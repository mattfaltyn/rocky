//! End-to-end coverage for `rocky compile --model` reporting scope.

use std::fs;
use std::process::{Command, Output};

fn write_model(dir: &std::path::Path, name: &str, sql: &str, depends_on: &[&str]) {
    fs::write(dir.join(format!("{name}.sql")), sql).expect("write model sql");
    let dependencies = depends_on
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        dir.join(format!("{name}.toml")),
        format!(
            "name = \"{name}\"\ndepends_on = [{dependencies}]\n\n[strategy]\ntype = \"full_refresh\"\n\n[target]\ncatalog = \"c\"\nschema = \"s\"\ntable = \"{name}\"\n"
        ),
    )
    .expect("write model sidecar");
}

fn compile(models: &std::path::Path, model: Option<&str>, output: &str, portable: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rocky"));
    command
        .arg("compile")
        .arg("--models")
        .arg(models)
        .arg("--output")
        .arg(output)
        .env("RUST_LOG", "error");
    if let Some(model) = model {
        command.arg("--model").arg(model);
    }
    if portable {
        command.arg("--target-dialect").arg("bq");
    }
    command.output().expect("spawn rocky compile")
}

fn parse_json(output: &Output) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).unwrap_or_else(|error| {
        panic!(
            "stdout is not JSON: {error}\nstdout: {stdout}\nstderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn selected_model_scopes_json_and_text_output() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let models = tmp.path().join("models");
    fs::create_dir(&models).expect("create models");
    write_model(&models, "upstream", "SELECT 1 AS id", &[]);
    write_model(&models, "selected", "SELECT 2 AS id", &["upstream"]);

    let json_output = compile(&models, Some("selected"), "json", false);
    assert!(json_output.status.success());
    let json = parse_json(&json_output);
    assert_eq!(json["models"], 1);
    assert_eq!(json["execution_layers"], 1);
    assert_eq!(json["has_errors"], false);
    assert_eq!(json["models_detail"].as_array().unwrap().len(), 1);
    assert_eq!(json["models_detail"][0]["name"], "selected");

    let text_output = compile(&models, Some("selected"), "table", false);
    assert!(text_output.status.success());
    let text = String::from_utf8(text_output.stdout).expect("utf8 stdout");
    assert!(text.contains("selected"), "selected model is shown: {text}");
    assert!(
        !text.contains("upstream"),
        "unselected model is hidden: {text}"
    );
    assert!(text.contains("Compiled: 1 models, 0 errors, 0 warnings"));
}

#[test]
fn unknown_model_is_an_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let models = tmp.path().join("models");
    fs::create_dir(&models).expect("create models");
    write_model(&models, "known", "SELECT 1 AS id", &[]);

    let output = compile(&models, Some("missing"), "json", false);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("model 'missing' not found (no transformation model with that name)")
    );
}

#[test]
fn unrelated_model_error_does_not_fail_selected_model() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let models = tmp.path().join("models");
    fs::create_dir(&models).expect("create models");
    write_model(&models, "selected", "SELECT 1 AS id", &[]);
    write_model(&models, "broken", "SELECT NVL(a, b) AS value FROM t", &[]);

    let output = compile(&models, Some("selected"), "json", true);
    assert!(output.status.success());
    let json = parse_json(&output);
    assert_eq!(json["has_errors"], false);
    assert!(json["diagnostics"].as_array().unwrap().is_empty());
    assert_eq!(json["models_detail"][0]["name"], "selected");
}

#[test]
fn selected_model_error_is_visible_and_fails() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let models = tmp.path().join("models");
    fs::create_dir(&models).expect("create models");
    write_model(&models, "clean", "SELECT 1 AS id", &[]);
    write_model(&models, "broken", "SELECT NVL(a, b) AS value FROM t", &[]);

    let output = compile(&models, Some("broken"), "json", true);
    assert!(!output.status.success());
    let json = parse_json(&output);
    assert_eq!(json["has_errors"], true);
    assert_eq!(json["models"], 1);
    assert!(
        json["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|diagnostic| diagnostic["code"] == "P001" && diagnostic["model"] == "broken")
    );
}

#[test]
fn no_selector_still_reports_the_whole_project() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let models = tmp.path().join("models");
    fs::create_dir(&models).expect("create models");
    write_model(&models, "upstream", "SELECT 1 AS id", &[]);
    write_model(&models, "downstream", "SELECT 2 AS id", &["upstream"]);

    let output = compile(&models, None, "json", false);
    assert!(output.status.success());
    let json = parse_json(&output);
    assert_eq!(json["models"], 2);
    assert_eq!(json["execution_layers"], 2);
    assert_eq!(json["models_detail"].as_array().unwrap().len(), 2);
    assert_eq!(json["has_errors"], false);
}
