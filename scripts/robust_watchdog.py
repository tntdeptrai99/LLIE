import subprocess
import time
import shutil
from pathlib import Path
import sys

def check_and_fix_corrupted_checkpoint(experiments_dir="experiments"):
    print(f"[Watchdog] Kiểm tra tính toàn vẹn của các file last.pt trong {experiments_dir}...")
    try:
        import torch
    except ImportError:
        print("[Watchdog] Không thể import torch để kiểm tra. Bỏ qua.")
        return

    count_fixed = 0
    # Quét tất cả các thư mục chứa last.pt
    for p in Path(experiments_dir).rglob("last.pt"):
        dir_path = p.parent
        best_path = dir_path / "best.pt"
        
        # Thử load file last.pt
        is_corrupted = False
        try:
            torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Watchdog] Phát hiện file checkpoint lỗi (corrupted): {p}")
            print(f"[Watchdog] Chi tiết lỗi: {e}")
            is_corrupted = True
            
        # Nếu lỗi và có file best.pt dự phòng thì tiến hành đè file
        if is_corrupted and best_path.exists():
            print(f"[Watchdog] Đang phục hồi {p.name} từ {best_path.name}...")
            try:
                shutil.copy(best_path, p)
                print(f"[Watchdog] Phục hồi thành công: {dir_path.name}")
                count_fixed += 1
            except Exception as copy_err:
                print(f"[Watchdog] Lỗi khi copy file phục hồi: {copy_err}")
                
    if count_fixed == 0:
        print("[Watchdog] Không phát hiện file last.pt nào cần phục hồi.")
    else:
        print(f"[Watchdog] Đã phục hồi thành công {count_fixed} file bị hỏng.")

def main():
    script_to_run = [sys.executable, "scripts/train_300e_suite.py"]
    
    print("[Watchdog] Bắt đầu khởi chạy bộ huấn luyện 300-epoch tự động phục hồi...")
    
    while True:
        try:
            print(f"\n[Watchdog] Running: {' '.join(script_to_run)}")
            # Chạy tiến trình chính
            result = subprocess.run(script_to_run, check=True)
            
            # Nếu chạy thành công và không bị văng lỗi
            if result.returncode == 0:
                print("\n[Watchdog] TIẾN TRÌNH HOÀN THÀNH XUẤT SẮC! Đã sinh xong toàn bộ báo cáo.")
                break
                
        except subprocess.CalledProcessError as e:
            print(f"\n[Watchdog] CẢNH BÁO: Tiến trình bị crash với mã lỗi (exit code) {e.returncode}!")
            print("[Watchdog] Bắt đầu quy trình tự động phục hồi (Auto-Recovery)...")
            
            # Kiểm tra và sửa lỗi file bị hỏng
            check_and_fix_corrupted_checkpoint("experiments/component_ablation")
            
            print("[Watchdog] Đang khởi động lại tiến trình sau 10 giây...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n[Watchdog] Đã nhận lệnh dừng từ người dùng. Kết thúc script.")
            break
            
if __name__ == "__main__":
    main()
