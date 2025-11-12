# hello.py
import datetime
import os

def main():
    name = os.getenv('GREETING_NAME', 'World')
    print(f"🎉 Hello, {name} from GitHub Actions!")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ The current time is: {current_time}")

if __name__ == "__main__":
    main()