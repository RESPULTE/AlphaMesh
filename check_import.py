import sys
import traceback

try:
    print("Test imported successfully.")
except BaseException:
    print("--- IMPORT ERROR ---")
    print(traceback.format_exc())
    sys.exit(1)
