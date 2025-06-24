import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

def simulate_bird_flock_signal(num_birds=30, base_v=20, lam=0.03,
                                A_range=(0.3, 0.7), noise_range=(0.2, 0.3),
                               micro_amp_range=(0.1, 0.3),
                                duration=1.0, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    for _ in range(num_birds):
        v = base_v + np.random.normal(0, 0.1)
        fd = (2 * v) / lam
        micro_amp = np.random.uniform(*micro_amp_range)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * np.random.uniform(0.2, 0.8) * t + np.random.rand() * 2 * np.pi))
        mod_fd = fd + micro_amp 
        A = np.random.uniform(*A_range)
        phi = np.random.uniform(0, 2 * np.pi)
        signal += envelope * A * np.cos(2 * np.pi * mod_fd * t + phi)
        # Add weak harmonic
        signal += 0.1 * A * np.cos(2 * np.pi * 2 * mod_fd * t + phi)

    noise_level = np.random.uniform(*noise_range)
    signal += np.random.normal(0, noise_level, t.shape)
    return t, signal


def simulate_drone_swarm_signal(num_drones=6, base_v=40, lam=0.03, 
                                A_range=(0.4, 0.9), noise_range=(0.1, 0.3),
                                duration=1.0, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    for _ in range(num_drones):
        v = base_v + np.random.uniform(-2, 2)
        fd = (2 * v) / lam
        A = np.random.uniform(*A_range)
        phi = np.random.uniform(0, 2 * np.pi)
        rotor_freq = np.random.uniform(20, 40)
        mod = 1 + 0.05 * np.sin(2 * np.pi * rotor_freq * t)
        phase_noise = np.cumsum(np.random.normal(0, 0.01, size=t.shape))
        signal += A * mod * np.cos(2 * np.pi * fd * t + phi + phase_noise)

    noise_level = np.random.uniform(*noise_range)
    signal += np.random.normal(0, noise_level, t.shape)
    return t, signal


def simulate_stealth_uav_signal(base_v=250, lam=0.03, A_range=(0.2, 0.7),
                                noise_range=(0.1, 0.3), duration=1.0, fs=1000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    v = base_v + np.random.uniform(-20, 20)
    fd = (2 * v) / lam
    A = np.random.uniform(*A_range)
    phi = np.random.uniform(0, 2 * np.pi)
    signal = A * np.cos(2 * np.pi * fd * t + phi)


    # Add transient burst
    if np.random.rand() > 0.6:
        center = np.random.randint(200, 800)
        burst = np.exp(-((np.arange(len(t)) - center) ** 2) / (2 * 20 ** 2))
        signal += 0.2 * burst * np.cos(2 * np.pi * fd * t + phi)

    # Apply occlusion mask
    # mask = np.ones_like(t)
    # num_dips = np.random.randint(1, 4)
    # for _ in range(num_dips):
    #     center = np.random.randint(int(0.2 * len(t)), int(0.8 * len(t)))
    #     width = np.random.randint(50, 150)
    #     dip = np.exp(-0.5 * ((np.arange(len(t)) - center) / width) ** 2)
    #     mask *= (1 - dip)

    # signal *= mask
    noise_level = np.random.uniform(*noise_range)
    signal += np.random.normal(0, noise_level, t.shape)
    return t, signal


def save_stft_grayscale(signal, fs, out_path):
    f, t_stft, Zxx = stft(signal, fs=fs, nperseg=256, noverlap=192)
    magnitude = np.abs(Zxx)
    db = np.clip(20 * np.log10(magnitude + 1e-10), -100, 0)

    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.pcolormesh(t_stft, f, db, shading='gouraud', cmap='gray')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def generate(path="data", samples=50, noise_multiplier=1):
    targets = {
        'Bird Flock': simulate_bird_flock_signal,
        'Drone Swarm': simulate_drone_swarm_signal,
        'Stealth UAV': simulate_stealth_uav_signal
    }

    fs = 1000

    for label, sim_func in targets.items():
        class_dir = os.path.join(path, label.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples):
            _, signal = sim_func()
            filename = os.path.join(class_dir, f"{label.replace(' ', '_').lower()}_{i}.png")
            save_stft_grayscale(signal, fs=fs, out_path=filename)
    print(f"Generation complete for {path} set. Spectrograms saved to:", path)

