#!/usr/bin/env sh
# One-time download of the local GGUF models into the persistent `ollama_models` volume (~40 GB).
# Re-running is safe: Ollama skips layers it already has. Override models via LOCAL_*_MODEL env vars.
set -e
cd "$(dirname "$0")/.."

# Mistral Small 3.1 ships a vision projector (878 MB) the app never uses. Loaded, it pushes part of the text model off
# a 16 GB GPU (8 % on the CPU: ~17 instead of ~50 tokens/s). The default editor is a text-only copy of the same weights:
# identical output, fully on the GPU.
EDITOR_SOURCE="hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_S"
EDITOR_TEXT="mistral-small-3.1-text:q4_k_s"

SCREEN="${LOCAL_SCREEN_MODEL:-hf.co/Qwen/Qwen3-8B-GGUF:Q5_K_M}"
ANALYSIS="${LOCAL_ANALYSIS_MODEL:-hf.co/Qwen/Qwen3-14B-GGUF:Q5_K_M}"
WRITER="${LOCAL_WRITER_MODEL:-$ANALYSIS}"
EDITOR="${LOCAL_GERMAN_EDITOR_MODEL:-$EDITOR_TEXT}"
HUMANIZER="${LOCAL_HUMANIZER_MODEL:-hf.co/jialinyyzz/humanizer:Q6_K}"
if [ "$EDITOR" = "$EDITOR_TEXT" ]; then EDITOR_PULL="$EDITOR_SOURCE"; else EDITOR_PULL="$EDITOR"; fi

docker compose up -d ollama
echo "Waiting for Ollama..."
until docker compose exec -T ollama ollama list >/dev/null 2>&1; do sleep 2; done

for MODEL in $(printf '%s\n' "$SCREEN" "$ANALYSIS" "$WRITER" "$EDITOR_PULL" "$HUMANIZER" | sort -u); do
  echo "==> ollama pull $MODEL"
  docker compose exec -T ollama ollama pull "$MODEL"
done

if [ "$EDITOR" = "$EDITOR_TEXT" ]; then
  echo "==> ollama create $EDITOR_TEXT (text-only copy of $EDITOR_SOURCE)"
  # keep the first FROM (the weights), drop the second (the vision projector) and comments
  docker compose exec -T ollama sh -c "ollama show --modelfile '$EDITOR_SOURCE' | awk '/^#/ {next} /^FROM / { if (seen++) next } {print}' > /tmp/Modelfile.text && ollama create '$EDITOR_TEXT' -f /tmp/Modelfile.text"
fi

echo
docker compose exec -T ollama ollama list
echo
echo "Done. Restart the app so it picks up the models:  docker compose up -d"
