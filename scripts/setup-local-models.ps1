# One-time download of the local GGUF models into the persistent `ollama_models` volume (~40 GB).
# Re-running is safe: Ollama skips layers it already has. Override models via LOCAL_*_MODEL env vars.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Mistral Small 3.1 ships a vision projector (878 MB) the app never uses. Loaded, it pushes part of the text model off
# a 16 GB GPU (8 % on the CPU: ~17 instead of ~50 tokens/s). The default editor is a text-only copy of the same weights:
# identical output, fully on the GPU.
$editorSource = "hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_S"
$editorText   = "mistral-small-3.1-text:q4_k_s"

$screen   = if ($env:LOCAL_SCREEN_MODEL) { $env:LOCAL_SCREEN_MODEL } else { "hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M" }
$analysis = if ($env:LOCAL_ANALYSIS_MODEL) { $env:LOCAL_ANALYSIS_MODEL } else { "hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M" }
$writer   = if ($env:LOCAL_WRITER_MODEL) { $env:LOCAL_WRITER_MODEL } else { $analysis }
$editor   = if ($env:LOCAL_GERMAN_EDITOR_MODEL) { $env:LOCAL_GERMAN_EDITOR_MODEL } else { $editorText }
$humanizer = if ($env:LOCAL_HUMANIZER_MODEL) { $env:LOCAL_HUMANIZER_MODEL } else { "hf.co/jialinyyzz/humanizer:Q6_K" }
$editorPull = if ($editor -eq $editorText) { $editorSource } else { $editor }

docker compose up -d ollama
Write-Host "Waiting for Ollama..."
do { Start-Sleep -Seconds 2; docker compose exec -T ollama ollama list *> $null } until ($LASTEXITCODE -eq 0)

foreach ($model in @($screen, $analysis, $writer, $editorPull, $humanizer) | Select-Object -Unique) {
    Write-Host "==> ollama pull $model"
    docker compose exec -T ollama ollama pull $model
    if ($LASTEXITCODE -ne 0) { throw "Pull failed for $model" }
}

if ($editor -eq $editorText) {
    Write-Host "==> ollama create $editorText (text-only copy of $editorSource)"
    # keep the first FROM (the weights), drop the second (the vision projector) and comments
    $script = 'ollama show --modelfile "{0}" | awk ''/^#/ {{next}} /^FROM / {{ if (seen++) next }} {{print}}'' > /tmp/Modelfile.text && ollama create "{1}" -f /tmp/Modelfile.text' -f $editorSource, $editorText
    docker compose exec -T ollama sh -c $script
    if ($LASTEXITCODE -ne 0) { throw "Creating $editorText failed" }
}

docker compose exec -T ollama ollama list
Write-Host "`nDone. Restart the app so it picks up the models:  docker compose up -d"
