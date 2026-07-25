//! `rocky ci-diff` — detect changed models between git refs and generate a structural diff report.
//!
//! Shells out to `git diff --name-status` to find `.sql`, `.rocky`, and `.toml`
//! sidecar files that changed between a base ref (default: `main`) and HEAD.
//! Compiles the current working tree to extract model schemas, then classifies
//! each changed model as added, modified, or removed and generates a structured
//! diff report in JSON and Markdown formats.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use tracing::debug;

use rocky_compiler::compile::{self, CompilerConfig};
use rocky_core::ci_diff::{
    ColumnChangeType, ColumnDiff, DiffResult, DiffSummary, ModelDiffStatus, format_diff_markdown,
    format_diff_table,
};
use rocky_core::models::Model;

use crate::output::{CiDiffOutput, print_json};

// ---------------------------------------------------------------------------
// Git integration
// ---------------------------------------------------------------------------

/// A file change detected by git between two refs.
#[derive(Debug, Clone)]
struct ChangedFile {
    /// Path relative to the repository root.
    path: String,
    /// Previous path for a Git rename or copy record.
    old_path: Option<String>,
    /// Git diff status: A (added), D (deleted), M (modified), R (renamed), etc.
    status: char,
}

/// A changed model keyed internally by its resolved target when both refs
/// compile, or by filename stem as a best-effort fallback.
#[derive(Debug, Default)]
struct ModelChange {
    /// Internal model name shown in the structural report.
    model_name: String,
    /// Schema-map key on the base ref.
    base_schema_name: Option<String>,
    /// Schema-map key on HEAD.
    head_schema_name: Option<String>,
}

impl ModelChange {
    fn status(&self) -> ModelDiffStatus {
        match (
            self.base_schema_name.is_some(),
            self.head_schema_name.is_some(),
        ) {
            (true, true) => ModelDiffStatus::Modified,
            (true, false) => ModelDiffStatus::Removed,
            (false, true) => ModelDiffStatus::Added,
            (false, false) => ModelDiffStatus::Unchanged,
        }
    }
}

/// Reject a `base_ref` that git would interpret as an option rather than a
/// revision.
///
/// `base_ref` is operator- or CI-supplied (commonly wired from a PR/branch
/// variable such as `${{ github.base_ref }}`). The diff helpers below pass it
/// to `git diff` / `git ls-tree` / `git show` as a positional argument, so a
/// value beginning with `-` could smuggle flags into those invocations
/// (argument injection). Refs that begin with `-` are never valid revisions,
/// and an empty ref is meaningless here, so rejecting both fully closes the
/// vector without constraining legitimate refs (`origin/main`, `HEAD~3`, a SHA).
pub(crate) fn validate_base_ref(base_ref: &str) -> Result<()> {
    if base_ref.is_empty() {
        anyhow::bail!("base ref must not be empty");
    }
    if base_ref.starts_with('-') {
        anyhow::bail!(
            "invalid base ref '{base_ref}': must not begin with '-' (git would read it as an option)"
        );
    }
    Ok(())
}

/// Run `git diff --name-status` between `base_ref` and HEAD to find changed files.
///
/// Uses three-dot syntax (`base...HEAD`) for merge-base semantics — this matches
/// what CI systems care about: changes since the branch diverged from the base,
/// not changes since the base's current tip.
fn git_changed_files(base_ref: &str) -> Result<Vec<ChangedFile>> {
    let output = Command::new("git")
        .args(["diff", "--name-status", &format!("{base_ref}...HEAD")])
        .output()
        .context("failed to run `git diff` — is git installed and is this a git repository?")?;

    if !output.status.success() {
        // Fall back to two-dot syntax if three-dot fails (e.g. shallow clone
        // without the base ref). This is less precise but better than failing.
        debug!(
            "three-dot git diff failed (exit {}), falling back to two-dot",
            output.status
        );
        let output = Command::new("git")
            .args(["diff", "--name-status", base_ref, "HEAD"])
            .output()
            .context("failed to run `git diff` with two-dot syntax")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git diff failed: {stderr}");
        }

        return parse_name_status(&output.stdout);
    }

    parse_name_status(&output.stdout)
}

/// Parse the output of `git diff --name-status`.
fn parse_name_status(raw: &[u8]) -> Result<Vec<ChangedFile>> {
    let text = String::from_utf8_lossy(raw);
    let mut files = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: "<status>\t<path>" (or "<status>\t<old>\t<new>" for renames)
        let mut parts = line.splitn(3, '\t');
        let status_str = parts.next().unwrap_or("");
        let first_path = parts.next().unwrap_or("");
        let status = status_str.chars().next().unwrap_or('M');
        let second_path = parts.next();
        let (path, old_path) = match (status, second_path) {
            ('R' | 'C', Some(new_path)) => (new_path, Some(first_path.to_string())),
            _ => (first_path, None),
        };

        if !path.is_empty() {
            files.push(ChangedFile {
                path: path.to_string(),
                old_path,
                status,
            });
        }
    }

    Ok(files)
}

/// Return the model stem for a changed path under the configured models root.
///
/// Model loading is flat, so nested config files such as `groups/*.toml` are
/// not model sidecars. Contract sidecars use `<stem>.contract.toml`.
fn model_stem(path: &str, models_rel: Option<&str>) -> Option<String> {
    let path = Path::new(path);
    let relative = match models_rel {
        Some(root) => path.strip_prefix(root).ok()?,
        None => path,
    };
    if models_rel.is_some() && relative.components().count() != 1 {
        return None;
    }

    let file_name = relative.file_name()?.to_str()?;
    let stem = if let Some(stem) = file_name.strip_suffix(".contract.toml") {
        stem
    } else {
        match relative.extension().and_then(|ext| ext.to_str()) {
            Some("sql" | "rocky" | "toml") => relative.file_stem()?.to_str()?,
            _ => return None,
        }
    };

    (!stem.is_empty() && stem != "_defaults" && stem != "rocky" && stem != "Cargo")
        .then(|| stem.to_string())
}

fn model_matches_path(model: &Model, path: &str, models_rel: Option<&str>) -> bool {
    let Some(stem) = model_stem(path, models_rel) else {
        return false;
    };
    let source_path = Path::new(&model.file_path);
    if source_path.file_stem().and_then(|value| value.to_str()) != Some(stem.as_str()) {
        return false;
    }

    match Path::new(path).extension().and_then(|ext| ext.to_str()) {
        Some("sql" | "rocky") => source_path.extension() == Path::new(path).extension(),
        Some("toml") => true,
        _ => false,
    }
}

fn record_model_side(
    changes: &mut HashMap<String, ModelChange>,
    key: String,
    model_name: &str,
    is_base: bool,
) {
    let change = changes.entry(key).or_default();
    if is_base {
        change.base_schema_name = Some(model_name.to_string());
        if change.model_name.is_empty() {
            change.model_name = model_name.to_string();
        }
    } else {
        change.head_schema_name = Some(model_name.to_string());
        change.model_name = model_name.to_string();
    }
}

fn record_resolved_side(
    changes: &mut HashMap<String, ModelChange>,
    models: &[Model],
    path: &str,
    models_rel: Option<&str>,
    is_base: bool,
) {
    for model in models
        .iter()
        .filter(|model| model_matches_path(model, path, models_rel))
    {
        let target = &model.config.target;
        let key = format!("{}.{}.{}", target.catalog, target.schema, target.table);
        record_model_side(changes, key, &model.config.name, is_base);
    }
}

/// Classify changed model files by their externally visible target identity.
///
/// Resolved pairing is only safe when both refs compiled. Otherwise the
/// classifier preserves the previous filename-based, conservative behavior.
fn classify_model_changes(
    files: &[ChangedFile],
    models_rel: Option<&str>,
    base_models: Option<&[Model]>,
    head_models: Option<&[Model]>,
) -> HashMap<String, ModelChange> {
    let mut changes = HashMap::new();

    if let (Some(base), Some(head)) = (base_models, head_models) {
        for file in files {
            if file.status == 'R'
                && let Some(old_path) = file.old_path.as_deref()
            {
                record_resolved_side(&mut changes, base, old_path, models_rel, true);
                record_resolved_side(&mut changes, head, old_path, models_rel, false);
            }
            record_resolved_side(&mut changes, base, &file.path, models_rel, true);
            record_resolved_side(&mut changes, head, &file.path, models_rel, false);
        }
        return changes;
    }

    let mut statuses = HashMap::new();
    for file in files {
        let Some(stem) = model_stem(&file.path, models_rel).or_else(|| {
            file.old_path
                .as_deref()
                .and_then(|path| model_stem(path, models_rel))
        }) else {
            continue;
        };
        let status = match file.status {
            'A' | 'C' => ModelDiffStatus::Added,
            'D' => ModelDiffStatus::Removed,
            // Keep the prior conservative result when either ref could not
            // compile: a rename is one modification at its new path.
            _ => ModelDiffStatus::Modified,
        };
        statuses
            .entry(stem)
            .and_modify(|existing| {
                if *existing == ModelDiffStatus::Modified {
                    *existing = status;
                }
            })
            .or_insert(status);
    }

    for (name, status) in statuses {
        let (base_schema_name, head_schema_name) = match status {
            ModelDiffStatus::Added => (None, Some(name.clone())),
            ModelDiffStatus::Removed => (Some(name.clone()), None),
            ModelDiffStatus::Modified => (Some(name.clone()), Some(name.clone())),
            ModelDiffStatus::Unchanged => (None, None),
        };
        changes.insert(
            name.clone(),
            ModelChange {
                model_name: name,
                base_schema_name,
                head_schema_name,
            },
        );
    }

    changes
}

// ---------------------------------------------------------------------------
// Schema extraction (current working tree)
// ---------------------------------------------------------------------------

/// Typed column from the compiler's type-check output.
#[derive(Debug, Clone)]
pub(crate) struct TypedColumn {
    pub(crate) name: String,
    pub(crate) data_type: String,
}

/// Project the compiler's `typed_models` map into the local `TypedColumn` shape
/// used by [`diff_columns`].
fn typed_columns_from_compile(
    result: &rocky_compiler::compile::CompileResult,
) -> HashMap<String, Vec<TypedColumn>> {
    let mut schemas = HashMap::new();
    for (model_name, typed_cols) in &result.type_check.typed_models {
        let cols: Vec<TypedColumn> = typed_cols
            .iter()
            .map(|tc| TypedColumn {
                name: tc.name.clone(),
                data_type: format!("{:?}", tc.data_type),
            })
            .collect();
        schemas.insert(model_name.clone(), cols);
    }
    schemas
}

/// Compile the models directory and return the full compile result.
///
/// `lineage-diff` needs the result's `semantic_graph` to compute downstream
/// traces; `ci-diff` only needs the per-model column schemas, which are
/// projected via [`typed_columns_from_compile`].
fn compile_head(
    models_dir: &Path,
    source_schemas: HashMap<String, Vec<rocky_compiler::types::TypedColumn>>,
) -> Result<rocky_compiler::compile::CompileResult> {
    let config = CompilerConfig {
        models_dir: models_dir.to_path_buf(),
        contracts_dir: None,
        source_schemas,
        source_column_info: HashMap::new(),
        ..Default::default()
    };

    compile::compile(&config).context("failed to compile models in the current working tree")
}

/// Try to compile the project as it stood at `base_ref` by checking out the
/// models directory at that ref into a temp directory and running the same
/// compile path as HEAD.
///
/// Returns:
/// - `Ok(result)` when the base ref's models compiled cleanly.
/// - `Err(reason)` with a short human-readable reason when the base could
///   not be materialized or did not compile. The reason is intended for the
///   semantic-diff gate's `BreakingChangesGateSkipped` audit event;
///   `compute_ci_diff` calls `.ok()` and treats `None` the same way the
///   pre-#510 `compile_base_ref` did.
///
/// `source_schemas` seeds the compile from the *current* warehouse cache,
/// not historical types. That's fine for diff purposes — typecheck on
/// historical models with today's leaf types still detects the model-level
/// schema drift that ci-diff is looking for, and there's no per-ref cache
/// to restore from.
pub fn extract_base_compile(
    base_ref: &str,
    models_dir: &Path,
    source_schemas: HashMap<String, Vec<rocky_compiler::types::TypedColumn>>,
) -> Result<rocky_compiler::compile::CompileResult, String> {
    let models_rel = match find_models_relative_path(models_dir) {
        Some(p) => p,
        None => {
            return Err("could not determine models directory relative to repo root".to_string());
        }
    };

    let tmp = match tempfile::tempdir() {
        Ok(t) => t,
        Err(e) => {
            return Err(format!(
                "failed to create temp dir for base extraction: {e}"
            ));
        }
    };

    let ls_output = Command::new("git")
        .args(["ls-tree", "-r", "--name-only", base_ref, &models_rel])
        .output();
    let ls_output = match ls_output {
        Ok(o) if o.status.success() => o,
        _ => {
            return Err(format!(
                "git ls-tree failed for base ref '{base_ref}' — models directory missing at that ref?"
            ));
        }
    };

    let file_list = String::from_utf8_lossy(&ls_output.stdout);
    let mut wrote_any = false;
    for file_path in file_list.lines() {
        let file_path = file_path.trim();
        if file_path.is_empty() {
            continue;
        }
        let rel = match file_path.strip_prefix(&models_rel) {
            Some(r) => r.trim_start_matches('/'),
            None => continue,
        };
        let dest = tmp.path().join(rel);
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let show_output = Command::new("git")
            .args(["show", &format!("{base_ref}:{file_path}")])
            .output();
        if let Ok(o) = show_output
            && o.status.success()
            && std::fs::write(&dest, &o.stdout).is_ok()
        {
            wrote_any = true;
        }
    }
    if !wrote_any {
        return Err(format!("no model files found at base ref '{base_ref}'"));
    }

    let config = CompilerConfig {
        models_dir: tmp.path().to_path_buf(),
        contracts_dir: None,
        source_schemas,
        source_column_info: HashMap::new(),
        ..Default::default()
    };

    compile::compile(&config).map_err(|e| format!("base ref '{base_ref}' did not compile: {e}"))
}

/// Find the models directory path relative to the git repo root.
fn find_models_relative_path(models_dir: &Path) -> Option<String> {
    let abs_models = std::fs::canonicalize(models_dir).ok()?;

    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let repo_root = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let repo_root = PathBuf::from(&repo_root);
    let repo_root = std::fs::canonicalize(&repo_root).ok()?;

    abs_models
        .strip_prefix(&repo_root)
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------
// Diff generation
// ---------------------------------------------------------------------------

/// Compare column schemas between base and head to produce column-level diffs.
fn diff_columns(base_cols: &[TypedColumn], head_cols: &[TypedColumn]) -> Vec<ColumnDiff> {
    let base_map: HashMap<&str, &str> = base_cols
        .iter()
        .map(|c| (c.name.as_str(), c.data_type.as_str()))
        .collect();
    let head_map: HashMap<&str, &str> = head_cols
        .iter()
        .map(|c| (c.name.as_str(), c.data_type.as_str()))
        .collect();

    let mut diffs = Vec::new();

    // Check for added or type-changed columns
    for col in head_cols {
        match base_map.get(col.name.as_str()) {
            None => {
                diffs.push(ColumnDiff {
                    column_name: col.name.clone(),
                    change_type: ColumnChangeType::Added,
                    old_type: None,
                    new_type: Some(col.data_type.clone()),
                });
            }
            Some(old_type) if *old_type != col.data_type.as_str() => {
                diffs.push(ColumnDiff {
                    column_name: col.name.clone(),
                    change_type: ColumnChangeType::TypeChanged,
                    old_type: Some(old_type.to_string()),
                    new_type: Some(col.data_type.clone()),
                });
            }
            _ => {}
        }
    }

    // Check for removed columns
    for col in base_cols {
        if !head_map.contains_key(col.name.as_str()) {
            diffs.push(ColumnDiff {
                column_name: col.name.clone(),
                change_type: ColumnChangeType::Removed,
                old_type: Some(col.data_type.clone()),
                new_type: None,
            });
        }
    }

    diffs
}

/// Build the full diff report from git changes and compiled schemas.
fn build_diff_results(
    model_changes: &HashMap<String, ModelChange>,
    head_schemas: &HashMap<String, Vec<TypedColumn>>,
    base_schemas: &HashMap<String, Vec<TypedColumn>>,
) -> Vec<DiffResult> {
    let mut results: Vec<DiffResult> = model_changes
        .values()
        .map(|change| {
            let status = change.status();
            let column_changes = match status {
                ModelDiffStatus::Modified => {
                    let base_cols = change
                        .base_schema_name
                        .as_ref()
                        .and_then(|name| base_schemas.get(name));
                    let head_cols = change
                        .head_schema_name
                        .as_ref()
                        .and_then(|name| head_schemas.get(name));
                    match (base_cols, head_cols) {
                        (Some(base), Some(head)) => diff_columns(base, head),
                        _ => vec![],
                    }
                }
                ModelDiffStatus::Added => {
                    // Show all columns as added for new models
                    change
                        .head_schema_name
                        .as_ref()
                        .and_then(|name| head_schemas.get(name))
                        .map(|cols| {
                            cols.iter()
                                .map(|c| ColumnDiff {
                                    column_name: c.name.clone(),
                                    change_type: ColumnChangeType::Added,
                                    old_type: None,
                                    new_type: Some(c.data_type.clone()),
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                }
                ModelDiffStatus::Removed => {
                    // Show all columns as removed for deleted models
                    change
                        .base_schema_name
                        .as_ref()
                        .and_then(|name| base_schemas.get(name))
                        .map(|cols| {
                            cols.iter()
                                .map(|c| ColumnDiff {
                                    column_name: c.name.clone(),
                                    change_type: ColumnChangeType::Removed,
                                    old_type: Some(c.data_type.clone()),
                                    new_type: None,
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                }
                ModelDiffStatus::Unchanged => vec![],
            };

            DiffResult {
                model_name: change.model_name.clone(),
                status,
                row_count_before: None,
                row_count_after: None,
                column_changes,
                sample_changed_rows: None,
            }
        })
        .collect();

    // Sort by model name for deterministic output
    results.sort_by(|a, b| a.model_name.cmp(&b.model_name));
    results
}

// ---------------------------------------------------------------------------
// Shared diff computation
// ---------------------------------------------------------------------------

/// Result of computing a CI diff.
///
/// `head_compile` is `None` when the models directory is missing or the
/// HEAD compile fails — callers (e.g. `rocky lineage-diff`) that need the
/// `semantic_graph` for downstream traces must handle that gracefully.
///
/// `base_compile` is similarly `None` when the base ref can't be checked
/// out into a temp dir or fails to compile. Both are kept on the data
/// struct so the `--semantic` path can lower them into [`ProjectIr`]
/// without re-running the compiler.
pub(crate) struct CiDiffData {
    pub(crate) summary: DiffSummary,
    pub(crate) results: Vec<DiffResult>,
    pub(crate) head_compile: Option<rocky_compiler::compile::CompileResult>,
    pub(crate) base_compile: Option<rocky_compiler::compile::CompileResult>,
    /// Total count of files git reported as changed between `base_ref` and
    /// HEAD (any extension, before the model-file filter). Lets callers
    /// distinguish "PR is empty" from "PR is non-empty but only touches
    /// non-model files".
    pub(crate) changed_file_count: usize,
}

/// Compute the CI diff between `base_ref` and HEAD without printing.
///
/// Shared between `run_ci_diff` and `run_lineage_diff` so the lineage-diff
/// command can enrich the per-column diff with downstream traces from
/// HEAD's `semantic_graph` without rerunning git or the compiler.
pub(crate) fn compute_ci_diff(
    config_path: &Path,
    state_path: &Path,
    base_ref: &str,
    models_dir: &Path,
    cache_ttl_override: Option<u64>,
) -> Result<CiDiffData> {
    validate_base_ref(base_ref)?;

    // Load cached source schemas once and seed both compiles (current
    // tree + base ref) with the same map so the resulting per-model
    // type diffs measure real schema drift rather than
    // `Unknown`-vs-`Unknown` noise. Degrades to empty when the cache is
    // cold or `[cache.schemas] enabled = false`.
    let source_schemas = match rocky_core::config::load_rocky_config(config_path) {
        Ok(cfg) => {
            let schema_cfg = cfg.cache.schemas.with_ttl_override(cache_ttl_override);
            crate::source_schemas::load_cached_source_schemas(&schema_cfg, state_path)
        }
        Err(_) => HashMap::new(),
    };

    let changed_files = git_changed_files(base_ref)?;
    let changed_file_count = changed_files.len();
    if changed_files.is_empty() {
        return Ok(CiDiffData {
            summary: DiffSummary {
                total_models: 0,
                unchanged: 0,
                modified: 0,
                added: 0,
                removed: 0,
            },
            results: vec![],
            head_compile: None,
            base_compile: None,
            changed_file_count,
        });
    }

    let models_rel = find_models_relative_path(models_dir);
    if classify_model_changes(&changed_files, models_rel.as_deref(), None, None).is_empty() {
        return Ok(CiDiffData {
            summary: DiffSummary {
                total_models: 0,
                unchanged: 0,
                modified: 0,
                added: 0,
                removed: 0,
            },
            results: vec![],
            head_compile: None,
            base_compile: None,
            changed_file_count,
        });
    }

    // Compile HEAD: keep the full result so callers can reach into
    // `semantic_graph`. Schema extraction below is a cheap projection.
    let head_compile = if models_dir.is_dir() {
        match compile_head(models_dir, source_schemas.clone()) {
            Ok(r) => Some(r),
            Err(e) => {
                debug!("HEAD compilation failed: {e}");
                None
            }
        }
    } else {
        None
    };
    let head_schemas = head_compile
        .as_ref()
        .map(typed_columns_from_compile)
        .unwrap_or_default();

    let base_compile = if models_dir.is_dir() {
        extract_base_compile(base_ref, models_dir, source_schemas).ok()
    } else {
        None
    };
    let base_schemas = base_compile
        .as_ref()
        .map(typed_columns_from_compile)
        .unwrap_or_default();

    let model_changes = classify_model_changes(
        &changed_files,
        models_rel.as_deref(),
        base_compile
            .as_ref()
            .map(|result| result.project.models.as_slice()),
        head_compile
            .as_ref()
            .map(|result| result.project.models.as_slice()),
    );
    let results = build_diff_results(&model_changes, &head_schemas, &base_schemas);
    let summary = DiffSummary::from_results(&results);

    Ok(CiDiffData {
        summary,
        results,
        head_compile,
        base_compile,
        changed_file_count,
    })
}

// ---------------------------------------------------------------------------
// Semantic breaking-change lowering
// ---------------------------------------------------------------------------

/// Lower a [`rocky_compiler::compile::CompileResult`] into a
/// [`rocky_ir::ProjectIr`] suitable for the
/// [`rocky_core::breaking_change`] classifier.
///
/// Each model in `result.project.models` is converted via
/// [`rocky_core::models::Model::to_model_ir`] (which leaves
/// `typed_columns` empty) and then enriched with the typed columns from
/// `result.type_check.typed_models`, keyed by `config.name`. Models the
/// type-checker did not produce columns for keep their empty
/// `typed_columns` vec — the classifier handles this gracefully (it
/// just won't emit per-column findings for that model).
///
/// `dag` and `lineage_edges` are left empty: the classifier ignores both
/// (they are implementation-detail fields per the
/// `rocky_core::breaking_change` module docs).
pub fn project_ir_from_compile(
    result: &rocky_compiler::compile::CompileResult,
) -> rocky_ir::ProjectIr {
    let typed = &result.type_check.typed_models;
    let models = result
        .project
        .models
        .iter()
        .map(|m| {
            let mut ir = m.to_model_ir();
            if let Some(cols) = typed.get(&m.config.name) {
                ir.typed_columns = cols.clone();
            }
            ir
        })
        .collect();
    rocky_ir::ProjectIr {
        models,
        dag: Vec::new(),
        lineage_edges: Vec::new(),
    }
}

/// Run the semantic breaking-change classifier across `base` and `head`
/// compiles. Returns an empty vec when either side is `None`.
fn semantic_findings(
    base: Option<&rocky_compiler::compile::CompileResult>,
    head: Option<&rocky_compiler::compile::CompileResult>,
) -> Vec<rocky_core::breaking_change::BreakingFinding> {
    match (base, head) {
        (Some(b), Some(h)) => {
            let old = project_ir_from_compile(b);
            let new = project_ir_from_compile(h);
            rocky_core::breaking_change::diff_project_ir(&old, &new)
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Public command entry point
// ---------------------------------------------------------------------------

/// Execute `rocky ci-diff`.
///
/// `semantic` enables the typed-IR breaking-change classifier
/// ([`rocky_core::breaking_change::diff_project_ir`]); findings are
/// attached to the JSON output under `breaking_findings`. The flag is
/// informational only: even a `Breaking` finding does not change the
/// exit code. The hard gate lives on `rocky branch promote`.
pub fn run_ci_diff(
    config_path: &Path,
    state_path: &Path,
    base_ref: &str,
    models_dir: &Path,
    output_json: bool,
    semantic: bool,
    cache_ttl_override: Option<u64>,
) -> Result<()> {
    let data = compute_ci_diff(
        config_path,
        state_path,
        base_ref,
        models_dir,
        cache_ttl_override,
    )?;

    if data.results.is_empty() && data.summary.total_models == 0 {
        // No model-level diff — distinguish "PR is empty" from "PR touched
        // non-model files only" the same way `rocky ci-diff` did before
        // the `compute_ci_diff` extraction.
        if output_json {
            let output = CiDiffOutput::new(
                base_ref.to_string(),
                "HEAD".to_string(),
                data.summary,
                vec![],
            );
            print_json(&output)?;
        } else if data.changed_file_count == 0 {
            println!("Rocky CI Diff ({base_ref}...HEAD)\n");
            println!("No changed model files detected.");
        } else {
            println!("Rocky CI Diff ({base_ref}...HEAD)\n");
            println!(
                "{} file(s) changed, but no model files (.sql, .rocky) were affected.",
                data.changed_file_count,
            );
        }
        return Ok(());
    }

    let findings = if semantic {
        semantic_findings(data.base_compile.as_ref(), data.head_compile.as_ref())
    } else {
        Vec::new()
    };

    if output_json {
        let output = CiDiffOutput::new(
            base_ref.to_string(),
            "HEAD".to_string(),
            data.summary,
            data.results,
        )
        .with_breaking_findings(findings);
        print_json(&output)?;
    } else {
        println!("Rocky CI Diff ({base_ref}...HEAD)\n");
        print!("{}", format_diff_table(&data.results));
        println!();
        println!("--- Markdown (for PR comment) ---\n");
        print!("{}", format_diff_markdown(&data.results));
        if semantic && !findings.is_empty() {
            println!();
            println!("--- Semantic Findings ({} total) ---\n", findings.len());
            for f in &findings {
                let sev = match f.severity {
                    rocky_core::breaking_change::BreakingSeverity::Breaking => "BREAKING",
                    rocky_core::breaking_change::BreakingSeverity::Warning => "WARNING",
                    rocky_core::breaking_change::BreakingSeverity::Info => "INFO",
                };
                println!("[{sev}] {:?}", f.change);
            }
        }
    }

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn changed(path: &str, status: char) -> ChangedFile {
        ChangedFile {
            path: path.to_string(),
            old_path: None,
            status,
        }
    }

    fn model_change(name: &str, status: ModelDiffStatus) -> ModelChange {
        let (base_schema_name, head_schema_name) = match status {
            ModelDiffStatus::Added => (None, Some(name.to_string())),
            ModelDiffStatus::Removed => (Some(name.to_string()), None),
            ModelDiffStatus::Modified => (Some(name.to_string()), Some(name.to_string())),
            ModelDiffStatus::Unchanged => (None, None),
        };
        ModelChange {
            model_name: name.to_string(),
            base_schema_name,
            head_schema_name,
        }
    }

    #[test]
    fn validate_base_ref_rejects_option_injection() {
        // A leading-dash ref would otherwise smuggle flags into the git
        // invocations (e.g. via ${{ github.base_ref }} in CI).
        assert!(validate_base_ref("--upload-pack=touch /tmp/pwned").is_err());
        assert!(validate_base_ref("-foo").is_err());
        assert!(validate_base_ref("").is_err());
        // Legitimate revisions still pass.
        assert!(validate_base_ref("origin/main").is_ok());
        assert!(validate_base_ref("HEAD~3").is_ok());
        assert!(validate_base_ref("abc1234").is_ok());
    }

    // -----------------------------------------------------------------------
    // parse_name_status
    // -----------------------------------------------------------------------

    #[test]
    fn parse_empty_output() {
        let files = parse_name_status(b"").unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn parse_added_modified_deleted() {
        let raw = b"A\tmodels/orders.sql\nM\tmodels/customers.sql\nD\tmodels/legacy.sql\n";
        let files = parse_name_status(raw).unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].status, 'A');
        assert_eq!(files[0].path, "models/orders.sql");
        assert_eq!(files[1].status, 'M');
        assert_eq!(files[1].path, "models/customers.sql");
        assert_eq!(files[2].status, 'D');
        assert_eq!(files[2].path, "models/legacy.sql");
    }

    #[test]
    fn parse_rename_preserves_both_paths() {
        let raw = b"R100\tmodels/old_name.sql\tmodels/new_name.sql\n";
        let files = parse_name_status(raw).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].status, 'R');
        assert_eq!(files[0].path, "models/new_name.sql");
        assert_eq!(files[0].old_path.as_deref(), Some("models/old_name.sql"));
    }

    #[test]
    fn parse_skips_blank_lines() {
        let raw = b"M\tmodels/foo.sql\n\n\nA\tmodels/bar.sql\n";
        let files = parse_name_status(raw).unwrap();
        assert_eq!(files.len(), 2);
    }

    // -----------------------------------------------------------------------
    // classify_model_changes
    // -----------------------------------------------------------------------

    #[test]
    fn classify_sql_files() {
        let files = vec![
            changed("models/orders.sql", 'A'),
            changed("models/customers.sql", 'M'),
            changed("models/legacy.sql", 'D'),
        ];
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["orders"].status(), ModelDiffStatus::Added);
        assert_eq!(changes["customers"].status(), ModelDiffStatus::Modified);
        assert_eq!(changes["legacy"].status(), ModelDiffStatus::Removed);
    }

    #[test]
    fn classify_rocky_files() {
        let files = vec![changed("models/pipeline.rocky", 'A')];
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["pipeline"].status(), ModelDiffStatus::Added);
    }

    #[test]
    fn classify_toml_sidecars() {
        let files = vec![changed("models/orders.toml", 'M')];
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["orders"].status(), ModelDiffStatus::Modified);
    }

    #[test]
    fn classify_ignores_non_model_files() {
        let files = vec![
            changed("rocky.toml", 'M'),
            changed("Cargo.toml", 'M'),
            changed("models/_defaults.toml", 'M'),
            changed("models/groups/orders.toml", 'M'),
            changed("README.md", 'M'),
            changed("src/main.rs", 'M'),
        ];
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert!(changes.is_empty());
    }

    #[test]
    fn classify_combined_sql_and_toml_prefers_significant() {
        // When both .sql (Added) and .toml (Modified) change for the same model,
        // the more significant status (Added) should win.
        let files = vec![
            changed("models/orders.toml", 'M'),
            changed("models/orders.sql", 'A'),
        ];
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["orders"].status(), ModelDiffStatus::Added);
    }

    #[test]
    fn classify_copy_as_added() {
        let files = parse_name_status(b"C100\tmodels/orders.sql\tmodels/purchases.sql\n").unwrap();
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["purchases"].status(), ModelDiffStatus::Added);
    }

    #[test]
    fn classify_rename_out_of_models_uses_old_path() {
        let files = parse_name_status(b"R100\tmodels/orders.sql\tarchive/orders.sql\n").unwrap();
        let changes = classify_model_changes(&files, Some("models"), None, None);
        assert_eq!(changes["orders"].status(), ModelDiffStatus::Modified);
    }

    // -----------------------------------------------------------------------
    // diff_columns
    // -----------------------------------------------------------------------

    #[test]
    fn diff_columns_no_changes() {
        let base = vec![
            TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            },
            TypedColumn {
                name: "name".into(),
                data_type: "VARCHAR".into(),
            },
        ];
        let diffs = diff_columns(&base, &base);
        assert!(diffs.is_empty());
    }

    #[test]
    fn diff_columns_added() {
        let base = vec![TypedColumn {
            name: "id".into(),
            data_type: "INT".into(),
        }];
        let head = vec![
            TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            },
            TypedColumn {
                name: "email".into(),
                data_type: "VARCHAR".into(),
            },
        ];
        let diffs = diff_columns(&base, &head);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].column_name, "email");
        assert_eq!(diffs[0].change_type, ColumnChangeType::Added);
        assert_eq!(diffs[0].new_type, Some("VARCHAR".into()));
    }

    #[test]
    fn diff_columns_removed() {
        let base = vec![
            TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            },
            TypedColumn {
                name: "legacy_flag".into(),
                data_type: "BOOLEAN".into(),
            },
        ];
        let head = vec![TypedColumn {
            name: "id".into(),
            data_type: "INT".into(),
        }];
        let diffs = diff_columns(&base, &head);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].column_name, "legacy_flag");
        assert_eq!(diffs[0].change_type, ColumnChangeType::Removed);
        assert_eq!(diffs[0].old_type, Some("BOOLEAN".into()));
    }

    #[test]
    fn diff_columns_type_changed() {
        let base = vec![TypedColumn {
            name: "price".into(),
            data_type: "FLOAT".into(),
        }];
        let head = vec![TypedColumn {
            name: "price".into(),
            data_type: "DOUBLE".into(),
        }];
        let diffs = diff_columns(&base, &head);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].column_name, "price");
        assert_eq!(diffs[0].change_type, ColumnChangeType::TypeChanged);
        assert_eq!(diffs[0].old_type, Some("FLOAT".into()));
        assert_eq!(diffs[0].new_type, Some("DOUBLE".into()));
    }

    #[test]
    fn diff_columns_mixed() {
        let base = vec![
            TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            },
            TypedColumn {
                name: "old_col".into(),
                data_type: "TEXT".into(),
            },
            TypedColumn {
                name: "amount".into(),
                data_type: "FLOAT".into(),
            },
        ];
        let head = vec![
            TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            },
            TypedColumn {
                name: "amount".into(),
                data_type: "DECIMAL".into(),
            },
            TypedColumn {
                name: "new_col".into(),
                data_type: "VARCHAR".into(),
            },
        ];
        let diffs = diff_columns(&base, &head);
        assert_eq!(diffs.len(), 3);

        let added: Vec<_> = diffs
            .iter()
            .filter(|d| d.change_type == ColumnChangeType::Added)
            .collect();
        let removed: Vec<_> = diffs
            .iter()
            .filter(|d| d.change_type == ColumnChangeType::Removed)
            .collect();
        let changed: Vec<_> = diffs
            .iter()
            .filter(|d| d.change_type == ColumnChangeType::TypeChanged)
            .collect();

        assert_eq!(added.len(), 1);
        assert_eq!(added[0].column_name, "new_col");
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].column_name, "old_col");
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].column_name, "amount");
    }

    // -----------------------------------------------------------------------
    // build_diff_results
    // -----------------------------------------------------------------------

    #[test]
    fn build_results_sorts_by_name() {
        let mut model_changes = HashMap::new();
        model_changes.insert(
            "c.s.zebra".into(),
            model_change("zebra", ModelDiffStatus::Added),
        );
        model_changes.insert(
            "c.s.alpha".into(),
            model_change("alpha", ModelDiffStatus::Modified),
        );

        let results = build_diff_results(&model_changes, &HashMap::new(), &HashMap::new());
        assert_eq!(results[0].model_name, "alpha");
        assert_eq!(results[1].model_name, "zebra");
    }

    #[test]
    fn build_results_with_schemas() {
        let mut model_changes = HashMap::new();
        model_changes.insert(
            "c.s.orders".into(),
            model_change("orders", ModelDiffStatus::Modified),
        );

        let mut base_schemas = HashMap::new();
        base_schemas.insert(
            "orders".into(),
            vec![
                TypedColumn {
                    name: "id".into(),
                    data_type: "INT".into(),
                },
                TypedColumn {
                    name: "price".into(),
                    data_type: "FLOAT".into(),
                },
            ],
        );

        let mut head_schemas = HashMap::new();
        head_schemas.insert(
            "orders".into(),
            vec![
                TypedColumn {
                    name: "id".into(),
                    data_type: "INT".into(),
                },
                TypedColumn {
                    name: "price".into(),
                    data_type: "DOUBLE".into(),
                },
                TypedColumn {
                    name: "tax".into(),
                    data_type: "DECIMAL".into(),
                },
            ],
        );

        let results = build_diff_results(&model_changes, &head_schemas, &base_schemas);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model_name, "orders");
        assert_eq!(results[0].status, ModelDiffStatus::Modified);
        assert_eq!(results[0].column_changes.len(), 2); // price type-changed + tax added
    }

    #[test]
    fn build_results_added_model_shows_all_columns() {
        let mut model_changes = HashMap::new();
        model_changes.insert(
            "c.s.new_model".into(),
            model_change("new_model", ModelDiffStatus::Added),
        );

        let mut head_schemas = HashMap::new();
        head_schemas.insert(
            "new_model".into(),
            vec![
                TypedColumn {
                    name: "id".into(),
                    data_type: "INT".into(),
                },
                TypedColumn {
                    name: "name".into(),
                    data_type: "VARCHAR".into(),
                },
            ],
        );

        let results = build_diff_results(&model_changes, &head_schemas, &HashMap::new());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].column_changes.len(), 2);
        assert!(
            results[0]
                .column_changes
                .iter()
                .all(|c| c.change_type == ColumnChangeType::Added)
        );
    }

    #[test]
    fn build_results_removed_model_shows_all_columns() {
        let mut model_changes = HashMap::new();
        model_changes.insert(
            "c.s.old_model".into(),
            model_change("old_model", ModelDiffStatus::Removed),
        );

        let mut base_schemas = HashMap::new();
        base_schemas.insert(
            "old_model".into(),
            vec![TypedColumn {
                name: "id".into(),
                data_type: "INT".into(),
            }],
        );

        let results = build_diff_results(&model_changes, &HashMap::new(), &base_schemas);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].column_changes.len(), 1);
        assert_eq!(
            results[0].column_changes[0].change_type,
            ColumnChangeType::Removed
        );
    }

    // -----------------------------------------------------------------------
    // project_ir_from_compile + semantic_findings (semantic mode stitching)
    // -----------------------------------------------------------------------

    use std::fs;
    use tempfile::TempDir;

    /// Write a minimal transformation model: `<name>.sql` + sidecar
    /// `<name>.toml`. Mirrors the test helper in `compile.rs`.
    fn write_model(dir: &Path, name: &str, sql: &str) {
        write_model_with_identity(dir, name, name, name, sql);
    }

    fn write_model_with_identity(
        dir: &Path,
        file_stem: &str,
        name: &str,
        target_table: &str,
        sql: &str,
    ) {
        let sql_path = dir.join(format!("{file_stem}.sql"));
        let toml_path = dir.join(format!("{file_stem}.toml"));
        fs::write(&sql_path, sql).unwrap();
        fs::write(
            &toml_path,
            format!(
                "name = \"{name}\"\n\n[strategy]\ntype = \"full_refresh\"\n\n[target]\ncatalog = \"c\"\nschema = \"s\"\ntable = \"{target_table}\"\n"
            ),
        )
        .unwrap();
    }

    fn write_inferred_model(dir: &Path, file_stem: &str, sql: &str) {
        fs::write(dir.join(format!("{file_stem}.sql")), sql).unwrap();
        fs::write(
            dir.join(format!("{file_stem}.toml")),
            "[strategy]\ntype = \"full_refresh\"\n\n[target]\ncatalog = \"c\"\nschema = \"s\"\n",
        )
        .unwrap();
    }

    /// Build a `HashMap<source_name, Vec<TypedColumn>>` to seed the
    /// compiler so SELECT FROM <source> yields concrete typed columns.
    fn source_schema(
        name: &str,
        cols: &[(&str, rocky_ir::RockyType)],
    ) -> HashMap<String, Vec<rocky_compiler::types::TypedColumn>> {
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            cols.iter()
                .map(|(n, t)| rocky_compiler::types::TypedColumn {
                    name: (*n).to_string(),
                    data_type: t.clone(),
                    nullable: true,
                })
                .collect(),
        );
        map
    }

    #[test]
    fn model_stem_normalizes_contract_and_rejects_nested_config() {
        assert_eq!(
            model_stem("models/orders.contract.toml", Some("models")).as_deref(),
            Some("orders")
        );
        assert_eq!(
            model_stem("models/groups/orders.toml", Some("models")),
            None
        );
        assert_eq!(
            model_stem("models/orders.sql", None).as_deref(),
            Some("orders")
        );
    }

    #[test]
    fn filename_inferred_rename_is_removed_and_added() {
        let sources = source_schema(
            "src_orders",
            &[
                ("id", rocky_ir::RockyType::Int64),
                ("status", rocky_ir::RockyType::String),
            ],
        );
        let base_dir = TempDir::new().unwrap();
        write_inferred_model(
            base_dir.path(),
            "orders",
            "SELECT id, status FROM src_orders",
        );
        let head_dir = TempDir::new().unwrap();
        write_inferred_model(
            head_dir.path(),
            "purchases",
            "SELECT id, status FROM src_orders",
        );
        let base_compile = compile_head(base_dir.path(), sources.clone()).expect("base compile");
        let head_compile = compile_head(head_dir.path(), sources).expect("head compile");
        let files = parse_name_status(
            b"R100\tmodels/orders.sql\tmodels/purchases.sql\n\
              R100\tmodels/orders.toml\tmodels/purchases.toml\n\
              R100\tmodels/orders.contract.toml\tmodels/purchases.contract.toml\n",
        )
        .unwrap();

        let changes = classify_model_changes(
            &files,
            Some("models"),
            Some(&base_compile.project.models),
            Some(&head_compile.project.models),
        );
        assert_eq!(changes.len(), 2);
        assert_eq!(changes["c.s.orders"].status(), ModelDiffStatus::Removed);
        assert_eq!(changes["c.s.purchases"].status(), ModelDiffStatus::Added);

        let results = build_diff_results(
            &changes,
            &typed_columns_from_compile(&head_compile),
            &typed_columns_from_compile(&base_compile),
        );
        let removed = results
            .iter()
            .find(|result| result.status == ModelDiffStatus::Removed)
            .unwrap();
        assert_eq!(removed.model_name, "orders");
        assert_eq!(removed.column_changes.len(), 2);
        assert!(
            removed
                .column_changes
                .iter()
                .all(|column| column.change_type == ColumnChangeType::Removed)
        );
        let added = results
            .iter()
            .find(|result| result.status == ModelDiffStatus::Added)
            .unwrap();
        assert_eq!(added.model_name, "purchases");
        assert_eq!(added.column_changes.len(), 2);
        assert!(
            added
                .column_changes
                .iter()
                .all(|column| column.change_type == ColumnChangeType::Added)
        );
    }

    #[test]
    fn rename_with_stable_target_is_modified_and_fallback_stays_conservative() {
        let sources = source_schema("src_orders", &[("id", rocky_ir::RockyType::Int64)]);
        let base_dir = TempDir::new().unwrap();
        write_model_with_identity(
            base_dir.path(),
            "orders_v1",
            "old_orders",
            "orders",
            "SELECT id FROM src_orders",
        );
        let head_dir = TempDir::new().unwrap();
        write_model_with_identity(
            head_dir.path(),
            "orders_v2",
            "new_orders",
            "orders",
            "SELECT id FROM src_orders",
        );
        let base_compile = compile_head(base_dir.path(), sources.clone()).expect("base compile");
        let head_compile = compile_head(head_dir.path(), sources).expect("head compile");
        let files = parse_name_status(
            b"R100\tmodels/orders_v1.sql\tmodels/orders_v2.sql\n\
              R100\tmodels/orders_v1.toml\tmodels/orders_v2.toml\n",
        )
        .unwrap();

        let changes = classify_model_changes(
            &files,
            Some("models"),
            Some(&base_compile.project.models),
            Some(&head_compile.project.models),
        );
        assert_eq!(changes.len(), 1);
        let change = &changes["c.s.orders"];
        assert_eq!(change.status(), ModelDiffStatus::Modified);
        assert_eq!(change.model_name, "new_orders");
        assert_eq!(change.base_schema_name.as_deref(), Some("old_orders"));
        assert_eq!(change.head_schema_name.as_deref(), Some("new_orders"));

        let results = build_diff_results(
            &changes,
            &typed_columns_from_compile(&head_compile),
            &typed_columns_from_compile(&base_compile),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, ModelDiffStatus::Modified);
        assert!(results[0].column_changes.is_empty());

        let fallback = classify_model_changes(
            &files,
            Some("models"),
            None,
            Some(&head_compile.project.models),
        );
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback["orders_v2"].status(), ModelDiffStatus::Modified);
    }

    #[test]
    fn contract_sidecar_add_or_delete_keeps_existing_model_modified() {
        let dir = TempDir::new().unwrap();
        write_model(
            dir.path(),
            "orders",
            "SELECT id FROM (SELECT 1 AS id) AS src",
        );
        let compile = compile_head(dir.path(), HashMap::new()).expect("compile");

        for status in ['A', 'D'] {
            let files = vec![changed("models/orders.contract.toml", status)];
            let changes = classify_model_changes(
                &files,
                Some("models"),
                Some(&compile.project.models),
                Some(&compile.project.models),
            );
            assert_eq!(changes.len(), 1);
            assert_eq!(changes["c.s.orders"].status(), ModelDiffStatus::Modified);
        }
    }

    #[test]
    fn added_sql_does_not_match_existing_same_stem_rocky_model() {
        let sources = source_schema("src_orders", &[("id", rocky_ir::RockyType::Int64)]);
        let base_dir = TempDir::new().unwrap();
        fs::write(
            base_dir.path().join("orders.rocky"),
            "from src_orders\nselect { id }\n",
        )
        .unwrap();
        let head_dir = TempDir::new().unwrap();
        fs::write(
            head_dir.path().join("orders.rocky"),
            "from src_orders\nselect { id }\n",
        )
        .unwrap();
        fs::write(
            head_dir.path().join("orders.sql"),
            "---toml\nname = \"orders_sql\"\ntarget = { catalog = \"c\", schema = \"s\", table = \"orders_sql\" }\n---\nSELECT id FROM src_orders\n",
        )
        .unwrap();
        let base_compile = compile_head(base_dir.path(), sources.clone()).expect("base compile");
        let head_compile = compile_head(head_dir.path(), sources).expect("head compile");

        let changes = classify_model_changes(
            &[changed("models/orders.sql", 'A')],
            Some("models"),
            Some(&base_compile.project.models),
            Some(&head_compile.project.models),
        );
        assert_eq!(changes.len(), 1);
        assert_eq!(changes["c.s.orders_sql"].status(), ModelDiffStatus::Added);
    }

    #[test]
    fn project_ir_from_compile_stitches_typed_columns_from_type_check() {
        let dir = TempDir::new().unwrap();
        let models_dir = dir.path();
        // SELECT FROM a seeded source so the typechecker produces real
        // typed columns; SELECT-without-FROM yields an empty schema.
        write_model(models_dir, "orders", "SELECT id, name FROM src_orders");

        let sources = source_schema(
            "src_orders",
            &[
                ("id", rocky_ir::RockyType::Int64),
                ("name", rocky_ir::RockyType::String),
            ],
        );
        let result = compile_head(models_dir, sources).expect("compile succeeds");
        let ir = project_ir_from_compile(&result);

        assert_eq!(ir.models.len(), 1);
        let model = &ir.models[0];
        assert_eq!(&*model.name, "orders");
        assert!(
            !model.typed_columns.is_empty(),
            "typed_columns must be stitched from type_check.typed_models",
        );
        let names: Vec<&str> = model
            .typed_columns
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(names.contains(&"id"));
        assert!(names.contains(&"name"));
    }

    #[test]
    fn semantic_findings_flag_column_drop_as_breaking() {
        // Compile two minimal projects that differ only by a dropped
        // column on a shared model; assert the classifier surfaces a
        // `column_dropped` finding with `breaking` severity via the
        // stitched IR.
        let sources = source_schema(
            "src_orders",
            &[
                ("id", rocky_ir::RockyType::Int64),
                ("legacy_flag", rocky_ir::RockyType::String),
            ],
        );

        let base_dir = TempDir::new().unwrap();
        write_model(
            base_dir.path(),
            "orders",
            "SELECT id, legacy_flag FROM src_orders",
        );
        let head_dir = TempDir::new().unwrap();
        write_model(head_dir.path(), "orders", "SELECT id FROM src_orders");

        let base_compile = compile_head(base_dir.path(), sources.clone()).expect("base compile");
        let head_compile = compile_head(head_dir.path(), sources).expect("head compile");

        let findings = semantic_findings(Some(&base_compile), Some(&head_compile));
        let dropped: Vec<_> = findings
            .iter()
            .filter(|f| {
                matches!(
                    f.change,
                    rocky_core::breaking_change::BreakingChange::ColumnDropped { .. }
                )
            })
            .collect();
        assert_eq!(
            dropped.len(),
            1,
            "expected exactly one column_dropped finding, got findings: {findings:?}",
        );
        assert!(
            dropped[0].is_breaking(),
            "column_dropped must surface as breaking severity",
        );
    }

    #[test]
    fn semantic_findings_empty_when_either_side_missing() {
        // No compile on either side → classifier is skipped, empty vec.
        // The CLI relies on `skip_serializing_if = "Vec::is_empty"` to
        // omit the field from JSON output in this case.
        assert!(semantic_findings(None, None).is_empty());
    }
}
