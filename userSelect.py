import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import subprocess
import numpy as np

# 전역 변수
cap = None
cap2 = None
is_playing = False
is_playing2 = False
fps = 30
video_width, video_height = 450, 350

# 사용자 데이터
users = [
    {"name": "I'm Baby", "image": "./assets/images/user1.png"},
    {"name": "MUSIC♥", "image": "./assets/images/user2.png"},
    {"name": "CAFFEINE", "image": "./assets/images/user3.png"},
    {"name": "HelloWorld!", "image": "./assets/images/user4.png"}
]

# 화면 전환 함수
def show_frame(frame):
    frame.tkraise()

# 프로필 선택 이벤트
def on_user_selected(user_name):
    selected_user_label.config(text=f"Welcome, {user_name}!")
    show_frame(screen2)

# 둥근 테두리 그리기 함수
def draw_rounded_border(canvas, x, y, width, height, radius, border_color, bg_color):
    canvas.create_arc(x, y, x + 2 * radius, y + 2 * radius, start=90, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x + width - 2 * radius, y, x + width, y + 2 * radius, start=0, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x, y + height - 2 * radius, x + 2 * radius, y + height, start=180, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x + width - 2 * radius, y + height - 2 * radius, x + width, y + height, start=270, extent=90, fill=border_color, outline=border_color)
    canvas.create_rectangle(x + radius, y, x + width - radius, y + height, fill=border_color, outline=border_color)
    canvas.create_rectangle(x, y + radius, x + width, y + height - radius, fill=border_color, outline=border_color)
    canvas.create_rectangle(x + radius, y + radius, x + width - radius, y + height - radius, fill=bg_color, outline=bg_color)

# Tkinter 초기화
root = tk.Tk()
root.title("Netflix-style User Selection")
root.geometry("1200x800")
root.configure(bg="black")

# 컨테이너 생성
container = tk.Frame(root)
container.pack(fill="both", expand=True)

container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# 화면 1: 프로필 선택 화면
screen1 = tk.Frame(container, bg="black")
screen1.grid(row=0, column=0, sticky="nsew")

logo_image = Image.open("assets/images/gross_logo.JPG").resize((150, 50))
logo_photo = ImageTk.PhotoImage(logo_image)

title_label = tk.Label(screen1, image=logo_photo, bg="black")
title_label.pack(pady=20)

profile_frame = tk.Frame(screen1, bg="black")
profile_frame.pack()

for index, user in enumerate(users):
    # 이미지 로드
    try:
        img = Image.open(user["image"]).resize((150, 150))
    except FileNotFoundError:
        img = Image.new("RGB", (150, 150), color="gray")
    user_img = ImageTk.PhotoImage(img)
    
    # 프로필 버튼 생성
    btn = tk.Button(
        profile_frame,
        image=user_img,
        text=user["name"],
        compound="top",
        font=("Comic Sans MS", 14, "bold"),
        bg="white",
        fg="black",
        command=lambda name=user["name"]: on_user_selected(name)
    )
    btn.image = user_img
    btn.grid(row=index // 2, column=index % 2, padx=20, pady=20)

# 화면 2: 선택 뒤 화면
screen2 = tk.Frame(container, bg="black")
screen2.grid(row=0, column=0, sticky="nsew")

selected_user_label = tk.Label(screen2, text="", font=("Comic Sans MS", 24), bg="black", fg="white")
selected_user_label.grid(row=0, column=0, columnspan=3, pady=10)

# 원본 동영상 영역
original_frame = tk.Frame(screen2, bg="black", width=video_width, height=video_height + 170)
original_frame.grid(row=1, column=0, padx=(50, 10), pady=10)

canvas = tk.Canvas(original_frame, width=video_width, height=video_height, bg="black", highlightthickness=0)
canvas.pack(pady=5)
draw_rounded_border(canvas, 0, 0, video_width, video_height, 20, "white", "black")

# 버튼 영역
original_controls = tk.Frame(original_frame, bg="black")
original_controls.pack(side="bottom", pady=10)

# Play 버튼
play_button = tk.Button(original_controls, text="Play", command=lambda: print("Play"))
play_button.grid(row=0, column=0, padx=5)

# Stop 버튼
stop_button = tk.Button(original_controls, text="Stop", command=lambda: print("Stop"))
stop_button.grid(row=0, column=1, padx=5)

# 뒤로 가기 버튼
back_button = tk.Button(screen2, text="Back to Profiles", command=lambda: show_frame(screen1), bg="white", fg="black")
back_button.grid(row=2, column=0, columnspan=3, pady=20)

# 초기 화면 설정
show_frame(screen1)

# 실행
root.mainloop()
