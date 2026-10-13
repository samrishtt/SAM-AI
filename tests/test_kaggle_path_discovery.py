import io, json, os, sys, tarfile, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(HERE, "..", "scripts"), HERE):
    if os.path.isfile(os.path.join(_cand, "kaggle_path_discovery.py")):
        sys.path.insert(0, _cand)
        break
import kaggle_path_discovery as k

def mk(p, text=""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)

def layout_a(root):  # extracted dataset + competitions/ + models/ (what Run 2 logs imply)
    mk(f"{root}/datasets/samrishb/sam-ai-source-bundle/sam_ai/__init__.py")
    mk(f"{root}/datasets/samrishb/sam-ai-source-bundle/LICENSE")
    mk(f"{root}/competitions/arc-prize-2026-arc-agi-3/environment_files/ab12/metadata.json", "{}")
    mk(f"{root}/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/x.whl")
    m = f"{root}/models/gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1/1"
    mk(f"{m}/config.json", json.dumps({"model_type":"qwen2","hidden_size":5120,"num_hidden_layers":48,
        "quantization_config":{"quant_method":"bitsandbytes","load_in_4bit":True,"bnb_4bit_quant_type":"nf4"}}))
    mk(f"{m}/model-00001.safetensors", "x")

def test_layout_a():
    with tempfile.TemporaryDirectory() as r:
        layout_a(r); out = k.discover_inputs(r, verbose=False)
        assert out["sam_ai_parent"].endswith("sam-ai-source-bundle")
        assert out["environment_files"].endswith("environment_files")
        assert out["base_model_config"]["num_hidden_layers"] == 48 and out["base_model_config"]["load_in_4bit"]

def test_flat_layout():  # old-style /kaggle/input/<slug>
    with tempfile.TemporaryDirectory() as r:
        mk(f"{r}/sam-ai-source-bundle/sam_ai/__init__.py")
        mk(f"{r}/arc-prize-2026-arc-agi-3/environment_files/g/metadata.json", "{}")
        m = f"{r}/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1"
        mk(f"{m}/config.json", "{}"); mk(f"{m}/w.safetensors")
        got = k.discover_inputs(r, verbose=False)["base_model_dir"]
        assert os.path.normpath(got) == os.path.normpath(m)  # os.walk uses "\\" on Windows

def test_tarball():
    with tempfile.TemporaryDirectory() as r, tempfile.TemporaryDirectory() as ex:
        layout_a(r)
        import shutil; shutil.rmtree(f"{r}/datasets")
        src = tempfile.mkdtemp(); mk(f"{src}/sam_ai/__init__.py", "")
        with tarfile.open(f"{r}/sam_ai_source_bundle.tar.gz","w:gz") as t: t.add(f"{src}/sam_ai", arcname="sam_ai")
        p = k.find_sam_ai_parent(r, extract_dir=ex)
        assert os.path.isfile(f"{p}/sam_ai/__init__.py")

def test_missing_raises_with_tree():
    with tempfile.TemporaryDirectory() as r:
        layout_a(r); import shutil; shutil.rmtree(f"{r}/competitions")
        try: k.discover_inputs(r, verbose=False)
        except RuntimeError as e:
            assert "environment_files" in str(e) and "datasets/" in str(e); return
        raise AssertionError("should have raised")

def test_no_wrong_model_substitution():
    with tempfile.TemporaryDirectory() as r:
        layout_a(r)
        m = f"{r}/models/other/llama-8b/transformers/v1"; mk(f"{m}/config.json","{}"); mk(f"{m}/w.safetensors")
        import shutil; shutil.rmtree(f"{r}/models/gowillgo")
        d, _ = k.find_base_model(r); assert d is None

def test_env_dir_without_metadata_rejected():
    with tempfile.TemporaryDirectory() as r:
        mk(f"{r}/c/environment_files/readme.txt"); assert k.find_environment_files(r) is None

def test_tar_traversal_rejected():
    with tempfile.TemporaryDirectory() as r, tempfile.TemporaryDirectory() as ex:
        with tarfile.open(f"{r}/sam_ai_source_bundle.tar.gz","w:gz") as t:
            ti = tarfile.TarInfo("../evil.py"); ti.size = 1; t.addfile(ti, io.BytesIO(b"x"))
        try: k.find_sam_ai_parent(r, extract_dir=ex)
        except RuntimeError as e: assert "outside" in str(e); return
        raise AssertionError("traversal accepted")

def test_tar_symlink_rejected():
    with tempfile.TemporaryDirectory() as r, tempfile.TemporaryDirectory() as ex:
        with tarfile.open(f"{r}/sam_ai_source_bundle.tar.gz","w:gz") as t:
            ti = tarfile.TarInfo("sam_ai/l"); ti.type = tarfile.SYMTYPE; ti.linkname="/etc/passwd"; t.addfile(ti)
        try: k.find_sam_ai_parent(r, extract_dir=ex)
        except RuntimeError as e: assert "non-regular" in str(e); return
        raise AssertionError("symlink accepted")

def test_missing_root():
    assert "does not exist" in k.tree("/nonexistent_xyz")
