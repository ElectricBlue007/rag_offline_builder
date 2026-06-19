from config.settings import ModelSettings
from core.embedding_engine import build_embedding_engine


def test_default_model_settings_use_bge_m3_and_gemma4():
    settings = ModelSettings()
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_cache_dir == "models/embeddings"
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "gemma4"


def test_local_hash_provider_remains_available_for_offline_tests():
    engine = build_embedding_engine(provider="local_hash")
    assert len(engine.embed_text("offline retrieval")) == 64
