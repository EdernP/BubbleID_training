import torch
import sys

def check_gpu_compatibility():
    """
    Vérifie si le système est compatible avec l'entraînement GPU et l'AMP.
    """
    print(f"Version de PyTorch : {torch.__version__}")
    print(f"Version de Python : {sys.version}")

    if not torch.cuda.is_available():
        print("\n❌ PyTorch n'a pas accès à un GPU compatible CUDA.")
        print("   Raisons possibles :")
        print("   1. Vous n'avez pas de carte graphique NVIDIA.")
        print("   2. Les drivers NVIDIA (avec CUDA) ne sont pas installés ou pas à jour.")
        print("   3. Vous avez installé la version 'cpu-only' de PyTorch.")
        return

    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"\n✅ GPU détecté : {gpu_name}")

    # Les GPU avec Tensor Cores sont optimisés pour l'AMP.
    # Cela inclut les architectures Turing, Ampere, Ada Lovelace, etc.
    # Les noms contiennent souvent "RTX", "T4", "A100", "V100", "H100".
    compatible_keywords = ["RTX", "Turing", "Ampere", "Volta", "Ada", "Hopper", "A100", "H100", "V100", "T4"]

    is_compatible = any(keyword.lower() in gpu_name.lower() for keyword in compatible_keywords)

    if is_compatible:
        print("✅ Votre GPU est parfaitement compatible avec la Précision Mixte Automatique (AMP).")
        print("   Vous pouvez activer l'option pour accélérer l'entraînement.")
    else:
        print("⚠️ Votre GPU pourrait ne pas être optimisé pour l'AMP (absence de Tensor Cores).")
        print("   L'activer pourrait ne pas apporter de gain de performance, voire ralentir l'entraînement.")

if __name__ == "__main__":
    check_gpu_compatibility()
