"""
run_all_analysis.py
====================
사용법 (KSEM_plot/ 디렉터리 안에서 실행):
  python run_all_analysis.py
  python run_all_analysis.py --only 2 4
"""
import argparse, importlib, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ksem_flux_config as cfg  # sys.path + 경로 자동 설정

SCRIPTS = {
    "1": ("ana1_count_flux_scatter",     "Count-Flux correlation"),
    "2": ("ana2_spe_count_threshold",    "SPE threshold derivation"),
    "3": ("ana3_noise_characterization", "Quiet noise characterization"),
    "4": ("ana4_count_response_profile", "Count response profile"),
    "5": ("ana5_electron_precursor",     "Electron precursor analysis"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(SCRIPTS.keys()))
    args = parser.parse_args()
    run_nums = args.only or list(SCRIPTS.keys())

    t_total = time.time()
    for num in run_nums:
        mod_name, desc = SCRIPTS[num]
        print(f"\n{'='*60}\n  [ana{num}] {desc}\n{'='*60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            mod.main()
            print(f"  -> done ({time.time()-t0:.1f}s)")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
    print(f"Output dir: {cfg.OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
