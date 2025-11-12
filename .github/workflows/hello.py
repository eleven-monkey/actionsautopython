# hello.py
import datetime

def main():
    print("🎉 Hello from GitHub Actions!")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ The current time is: {current_time}")

if __name__ == "__main__":
    main()