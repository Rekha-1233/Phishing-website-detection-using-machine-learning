import os

db_path = "database.db"

if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print("✅ database.db deleted successfully.")
    except PermissionError:
        print("❌ File is locked. Please close all applications using it.")
else:
    print("ℹ️ database.db not found.")
