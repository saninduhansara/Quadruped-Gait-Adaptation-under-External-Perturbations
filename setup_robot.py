"""
Setup script to download the Unitree A1 robot model from DeepMind MuJoCo Menagerie.
"""
import os
import urllib.request
import zipfile

def setup_robot_model():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    a1_dir = os.path.join(model_dir, "unitree_a1")
    
    if os.path.exists(os.path.join(a1_dir, "a1.xml")):
        print(f"[OK] Unitree A1 model already exists at: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")
    
    print("[INFO] Cloning Google DeepMind MuJoCo Menagerie (Unitree A1)...")
    # Clone minimal menagerie or download zip
    import subprocess
    cmd = [
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "https://github.com/google-deepmind/mujoco_menagerie.git",
        os.path.join(model_dir, "menagerie_tmp")
    ]
    try:
        subprocess.run(cmd, check=True)
        sparse_cmd = ["git", "-C", os.path.join(model_dir, "menagerie_tmp"), "sparse-checkout", "set", "unitree_a1"]
        subprocess.run(sparse_cmd, check=True)
        import shutil
        shutil.move(os.path.join(model_dir, "menagerie_tmp", "unitree_a1"), a1_dir)
        shutil.rmtree(os.path.join(model_dir, "menagerie_tmp"))
        print(f"[SUCCESS] Unitree A1 model ready at: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")
    except Exception as e:
        print(f"[FALLBACK] Git sparse checkout failed: {e}. Downloading repository archive...")
        zip_url = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
        zip_path = os.path.join(model_dir, "menagerie.zip")
        urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if "unitree_a1/" in member:
                    zip_ref.extract(member, model_dir)
        import shutil
        extracted_path = os.path.join(model_dir, "mujoco_menagerie-main", "unitree_a1")
        if os.path.exists(extracted_path):
            shutil.move(extracted_path, a1_dir)
            shutil.rmtree(os.path.join(model_dir, "mujoco_menagerie-main"))
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"[SUCCESS] Downloaded Unitree A1 model to: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")

if __name__ == "__main__":
    setup_robot_model()
