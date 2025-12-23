# src/logger.py
import logging
import sys
import os
import traceback

def setup_logging():
    """
    設定 Logging 系統：
    1. 找出適合存放 Log 的路徑 (執行檔旁)
    2. 設定同時輸出到 檔案 與 Console
    3. 掛載全域例外攔截 (Global Exception Hook)
    """
    
    # 決定 Log 檔案路徑
    # 如果是打包後的 exe，存在 exe 旁邊
    # 如果是開發模式，存在專案根目錄
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        # src/logger.py 的上上層 (專案根目錄)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    log_file_path = os.path.join(base_dir, "editor_debug.log")

    # 設定 Log 格式與處理器
    # mode='w' 表示每次重開程式都清空舊 Log (如果你想保留歷史，改成 'a')
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # 讓終端機也能看到
        ]
    )
    # 這段代碼保證了：即使程式當機，錯誤訊息也會被寫入 Log 檔
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught Exception (Crash Detected):", exc_info=(exc_type, exc_value, exc_traceback))
        

    sys.excepthook = handle_exception

    logging.info(f"Logger initialized. Log file: {log_file_path}")
    print(f"[Logger] Debug file path: {log_file_path}")