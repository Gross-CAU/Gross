import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import numpy as np
import os

def draw_rounded_frame(canvas, x, y, width, height, radius, border_color, border_width, bg_color):
    """
    Draw a rounded rectangle frame with a transparent center.
    - canvas: The Canvas object.
    - x, y: Top-left corner of the rectangle.
    - width, height: Dimensions of the rectangle.
    - radius: Corner radius.
    - border_color: Color of the border.
    - border_width: Width of the border.
    - bg_color: Background color inside the rounded frame.
    """
    # Rounded corners
    canvas.create_arc(
        x, y, x + 2 * radius, y + 2 * radius,
        start=90, extent=90, fill=border_color, outline=border_color
    )  # Top-left
    canvas.create_arc(
        x + width - 2 * radius, y, x + width, y + 2 * radius,
        start=0, extent=90, fill=border_color, outline=border_color
    )  # Top-right
    canvas.create_arc(
        x, y + height - 2 * radius, x + 2 * radius, y + height,
        start=180, extent=90, fill=border_color, outline=border_color
    )  # Bottom-left
    canvas.create_arc(
        x + width - 2 * radius, y + height - 2 * radius, x + width, y + height,
        start=270, extent=90, fill=border_color, outline=border_color
    )  # Bottom-right

    # Top and bottom rectangles (between corners)
    canvas.create_rectangle(
        x + radius, y, x + width - radius, y + border_width,
        fill=border_color, outline=border_color
    )
    canvas.create_rectangle(
        x + radius, y + height - border_width, x + width - radius, y + height,
        fill=border_color, outline=border_color
    )

    # Left and right rectangles (between corners)
    canvas.create_rectangle(
        x, y + radius, x + border_width, y + height - radius,
        fill=border_color, outline=border_color
    )
    canvas.create_rectangle(
        x + width - border_width, y + radius, x + width, y + height - radius,
        fill=border_color, outline=border_color
    )

    # Inner background rectangle
    canvas.create_rectangle(
        x + border_width, y + border_width,
        x + width - border_width, y + height - border_width,
        fill=bg_color, outline=bg_color
    )


# 둥근 테두리 추가
def apply_rounded_frame_to_canvas(canvas, x, y, width, height, radius, border_color="white", border_width=10, bg_color="black"):
    """
    Wrapper to draw a rounded frame over an existing Canvas to simulate rounded edges.
    """
    draw_rounded_frame(canvas, x, y, width, height, radius, border_color, border_width, bg_color)


# 버튼을 클릭하여 Entry 위젯의 텍스트 가져오기 예제
def get_entry_text():
    input_text = entry_box.get()
    print("입력한 텍스트:", input_text)



def select_video():
    global cap, is_playing, fps
    video_path = filedialog.askopenfilename()
    if video_path:
        global current_vid 
        current_vid= video_path
        print(current_vid)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상의 fps 가져오기
        is_playing = True
        update_frame()  # 선택한 동영상 재생

def select_additional_video():
    global cap2, is_playing2, fps
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if video_path:
        cap2 = cv2.VideoCapture(video_path)
        fps = cap2.get(cv2.CAP_PROP_FPS)  # 동영상의 fps 가져오기
        is_playing2 = True
        update_frame2()  # 추가 동영상 재생

def round_corners(frame, radius):
    """Apply rounded corners to the video frame."""
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)
    rounded_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return rounded_frame

def update_frame():
    global cap, is_playing
    if cap is not None and is_playing:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (video_width, video_height))
            frame = round_corners(frame, 50)  # 둥근 모서리 적용
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk
            
            delay = int(1000 / fps)
            root.after(delay, update_frame)
        else:
            cap.release()
            is_playing = False  # 재생 중지

def update_frame2():
    global cap2, is_playing2
    if cap2 is not None and is_playing2:
        ret, frame = cap2.read()
        if ret:
            frame = cv2.resize(frame, (video_width, video_height))
            frame = round_corners(frame, 50)  # 둥근 모서리 적용
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas2.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas2.image = imgtk
            
            delay = int(1000 / fps)
            root.after(delay, update_frame2)
        else:
            cap2.release()
            is_playing2 = False


def play_video():
    global is_playing, cap, current_vid
    if current_vid:  # 동영상 파일이 선택되었는지 확인
        if cap is None or not cap.isOpened():  # cap이 해제되었거나 유효하지 않은 경우
            cap = cv2.VideoCapture(current_vid)  # 새로운 VideoCapture 객체 생성
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상을 처음 프레임으로 이동
        is_playing = True
        update_frame()

def play_additional_video():
    global is_playing2, cap2
    video_path = os.path.join(script_dir,"outputs/audio/fish_blackbox.mp4")
    cap2 = cv2.VideoCapture(video_path)
    if cap2 is not None and not is_playing2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_playing2 = True
        update_frame2()

def stop_video():
    global is_playing
    if cap is not None and is_playing:
        is_playing = False

def stop_additional_video():
    global is_playing2
    if cap2 is not None and is_playing2:
        is_playing2 = False

def run_external_script():
    global current_vid,cap2, is_playing2, fps
    show_loading()
    subprocess.run(["python",os.path.join(script_dir,"main.py"),current_vid,filter_type,text_prompt])
    show_finish()
    video_path = os.path.join(script_dir,"outputs/audio/fish_blackbox.mp4")
    if video_path:
        cap2 = cv2.VideoCapture(video_path)
        fps = cap2.get(cv2.CAP_PROP_FPS)  # 동영상의 fps 가져오기
        is_playing2 = False
    
    
def draw_rounded_border(canvas, x, y, width, height, radius, border_color, bg_color):
    """Create a rounded rectangle on a canvas."""
    canvas.create_arc(x, y, x + 2 * radius, y + 2 * radius, start=90, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x + width - 2 * radius, y, x + width, y + 2 * radius, start=0, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x, y + height - 2 * radius, x + 2 * radius, y + height, start=180, extent=90, fill=border_color, outline=border_color)
    canvas.create_arc(x + width - 2 * radius, y + height - 2 * radius, x + width, y + height, start=270, extent=90, fill=border_color, outline=border_color)
    canvas.create_rectangle(x + radius, y, x + width - radius, y + height, fill=border_color, outline=border_color)
    canvas.create_rectangle(x, y + radius, x + width, y + height - radius, fill=border_color, outline=border_color)
    canvas.create_rectangle(x + radius, y + radius, x + width - radius, y + height - radius, fill=bg_color, outline=bg_color)

def show_loading():
    # 캔버스 초기화
    canvas2.delete("all")
    draw_rounded_border(canvas2, 0, 0, video_width, video_height, 20, "white", "black")

    # "Loading..." 텍스트를 캔버스 중앙에 표시
    canvas2.create_text(
        video_width // 2,  # 캔버스 중앙 X 좌표
        video_height // 2,  # 캔버스 중앙 Y 좌표
        text="Loading...",  # 표시할 텍스트
        fill="white",  # 텍스트 색상
        font=("Arial", 20, "bold")  # 폰트와 크기
    )
    # UI 즉시 업데이트 강제
    canvas2.update_idletasks()

def show_finish():
    # 캔버스 초기화
    canvas2.delete("all")
    draw_rounded_border(canvas2, 0, 0, video_width, video_height, 20, "white", "black")

    # "Loading..." 텍스트를 캔버스 중앙에 표시
    canvas2.create_text(
        video_width // 2,  # 캔버스 중앙 X 좌표
        video_height // 2,  # 캔버스 중앙 Y 좌표
        text="Filtering Finished!",  # 표시할 텍스트
        fill="white",  # 텍스트 색상
        font=("Arial", 20, "bold")  # 폰트와 크기
    )
    # UI 즉시 업데이트 강제
    canvas2.update_idletasks()
    

filter_type = "none"
text_prompt = ""
def save_text_prompt():
    global text_prompt, current_user
    for user in users:  # users 리스트를 순회
        if user["name"] == current_user:  # user_name과 일치하는 사용자 찾기
            if "pred" in user:  # pred 키가 존재하는지 확인
                # pred 값을 text_prompt에 추가
                text_prompt = f"{entry_box.get()}{user['pred']}"
            else:
                # pred 값이 없는 경우 텍스트만 저장
                text_prompt = entry_box.get()
            break  # 사용자를 찾았으므로 루프 종료
    print(f"Text prompt saved: {text_prompt}")  # 저장된 text_prompt 출력


def set_filter_type(new_filter_type, selected_button):
    global filter_type
    filter_type = new_filter_type
    print(f"Filter type set to: {filter_type}")  # 현재 필터 유형 출력

    # 모든 버튼의 기본 스타일 초기화
    none_button.config(bg="lightgray")
    sticker_button.config(bg="lightgray")
    real_button.config(bg="lightgray")

    # 선택된 버튼의 스타일 변경
    selected_button.config(bg="#4a4a4a")  # 선택된 버튼을 강조하기 위해 배경색 변경
    
users = [
    {"name": "I'm Baby", "image": "assets/images/baby.png","pred" : "knife . gun . smoke . alchohol ."},
    {"name": "NO CLOWN", "image": "assets/images/sad.PNG","pred" : "clown ."},
    {"name": "Sad Fish", "image": "assets/images/user1.jpg"},
    {"name": "CookingFoil", "image": "assets/images/students.png"}
]

current_user =""

# 화면 전환 함수
def show_frame(frame):
    frame.tkraise()

# 프로필 선택 이벤트
def on_user_selected(user_name):
    global current_user
    current_user = user_name
    for user in users:  # users 리스트를 순회
        if user["name"] == user_name:  # user_name과 일치하는 사용자 찾기
            if "pred" in user:  # pred 키가 존재하는지 확인
                text_label.config(text="Predefined Object: "+user["pred"])
            else:
                text_label.config(text="")
            break  # 사용자를 찾았으므로 루프 종료
    show_frame(screen2)

    
# Tkinter 설정
root = tk.Tk()
root.title("GROSS!")
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

script_dir = os.path.dirname(os.path.abspath(__file__))


logo_image = Image.open(os.path.join(script_dir,"assets/images/gross_logo.JPG")).resize((150, 50))  # 로고 이미지 크기 조정
logo_photo = ImageTk.PhotoImage(logo_image)

title_label = tk.Label(screen1, image=logo_photo, bg="black")
title_label.pack(pady=20)

profile_frame = tk.Frame(screen1, bg="black")
profile_frame.pack()

for index, user in enumerate(users):
    # 이미지 로드
    try:
        img = Image.open(os.path.join(script_dir,user["image"])).resize((150, 150))
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
    
screen2 = tk.Frame(container, bg="black")
screen2.grid(row=0,column=0,sticky="nsew")


# 열 크기 균등 분배
screen2.grid_columnconfigure(0, weight=1)  # 첫 번째 열
screen2.grid_columnconfigure(1, weight=1)  # 두 번째 열
screen2.grid_columnconfigure(2, weight=1)  # 세 번째 열

# 로고 이미지 로드 및 중앙 배치


play_image = Image.open(os.path.join(script_dir,"assets/images/play.JPG")).resize((50, 50))  # Play 버튼 이미지 크기 조정
play_photo = ImageTk.PhotoImage(play_image)

stop_image = Image.open(os.path.join(script_dir,"assets/images/pause.JPG")).resize((50, 50))  # Stop 버튼 이미지 크기 조정
stop_photo = ImageTk.PhotoImage(stop_image)

select_image = Image.open(os.path.join(script_dir,"assets/images/upload.png")).resize((150, 50))  # Stop 버튼 이미지 크기 조정
select_photo = ImageTk.PhotoImage(select_image)

search_image = Image.open(os.path.join(script_dir,"assets/images/search.png")).resize((50, 50))  # 검색 버튼 이미지 크기 조정
search_photo = ImageTk.PhotoImage(search_image)

capy_real_image = Image.open(os.path.join(script_dir,"assets/images/realistic.png")).resize((50, 50))  # 검색 버튼 이미지 크기 조정
capy_real_photo = ImageTk.PhotoImage(capy_real_image)

capy_sticker_image = Image.open(os.path.join(script_dir,"assets/images/sticker.png")).resize((50, 50))  # 검색 버튼 이미지 크기 조정
capy_sticker_photo = ImageTk.PhotoImage(capy_sticker_image)

capy_mosaic_image = Image.open(os.path.join(script_dir,"assets/images/realistic_mosaic.png")).resize((50, 50))  # 검색 버튼 이미지 크기 조정
capy_mosaic_photo = ImageTk.PhotoImage(capy_mosaic_image)


logo_button = tk.Button(screen2, image=logo_photo, bg="black",borderwidth=0,activebackground="black",highlightthickness=0,command=lambda: show_frame(screen1))  # 텍스트 대신 이미지 사용
logo_button.grid(row=0, column=0, columnspan=3, pady=10)  # 화면 중앙 상단에 배치

# 원본 동영상 영역
video_width, video_height = 450, 350
original_frame = tk.Frame(screen2, bg="black", width=video_width, height=video_height + 170)  # 추가 공간 확보
original_frame.grid(row=1, column=0, padx=(50, 10), pady=10)  # 좌측 패딩 추가로 균형 조정
original_frame.pack_propagate(False)  # Frame 크기 고정

# 동영상 제목 레이블
original_label = tk.Label(original_frame, text="Original Video", font=("Comic Sans MS", 24, "bold"), bg="black", fg="white")
original_label.pack(side="top", pady=5)

# Canvas 생성 (동영상 영역)
canvas = tk.Canvas(original_frame, width=video_width, height=video_height, bg="black", highlightthickness=0)
canvas.pack(pady=5)  # Canvas 상하 여백 조정
draw_rounded_border(canvas, 0, 0, video_width, video_height, 20, "white", "black")

# 재생/정지 버튼 영역 (Original Video)
original_controls = tk.Frame(original_frame, bg="black")
original_controls.pack(side="bottom", pady=10)

# Play 버튼 (이미지)
play_button1 = tk.Button(original_controls, image=play_photo, bg="black", borderwidth=0, activebackground="black",highlightthickness=0,  # 클릭 시 배경색 유지
    activeforeground="black",command=play_video)
play_button1.grid(row=0, column=0, padx=5)

# Stop 버튼 (이미지)
stop_button1 = tk.Button(original_controls, image=stop_photo, bg="black", borderwidth=0, activebackground="black",highlightthickness=0,  # 클릭 시 배경색 유지
    activeforeground="black",command=stop_video)
stop_button1.grid(row=0, column=1, padx=5)
# Select Video 버튼 (가운데 배치)
select_button1 = tk.Button(original_controls, image=select_photo, bg="black", borderwidth=0, activebackground="black",highlightthickness=0,  # 클릭 시 배경색 유지
    activeforeground="black",command=select_video)
select_button1.grid(row=0, column=2, padx=5)
# 필터링된 동영상 영역
filtered_frame = tk.Frame(screen2, bg="black", width=video_width, height=video_height + 170)  # 추가 공간 확보
filtered_frame.grid(row=1, column=2, padx=(10, 50), pady=10)  # 우측 패딩 추가로 균형 조정
filtered_frame.pack_propagate(False)  # Frame 크기 고정

# 필터링된 동영상 제목 레이블
filtered_label = tk.Label(filtered_frame, text="Filtered Video", font=("Comic Sans MS", 24, "bold"), bg="black", fg="white")
filtered_label.pack(side="top", pady=5)

# Canvas 생성 (필터링된 동영상 영역)
canvas2 = tk.Canvas(filtered_frame, width=video_width, height=video_height, bg="black", highlightthickness=0)
canvas2.pack(pady=5)  # Canvas 상하 여백 조정
draw_rounded_border(canvas2, 0, 0, video_width, video_height, 20, "white", "black")

# 재생/정지 버튼 영역 (Filtered Video)
filtered_controls = tk.Frame(filtered_frame, bg="black")
filtered_controls.pack(side="bottom", pady=10)

# Play 버튼 (이미지)
play_button2 = tk.Button(filtered_controls, image=play_photo, bg="black", borderwidth=0, activebackground="black",highlightthickness=0,  # 클릭 시 배경색 유지
    activeforeground="black",command=play_additional_video)
play_button2.grid(row=0, column=0, padx=5)

# Stop 버튼 (이미지)
stop_button2 = tk.Button(filtered_controls, image=stop_photo, bg="black", borderwidth=0, activebackground="black",highlightthickness=0,  # 클릭 시 배경색 유지
    activeforeground="black",command=stop_additional_video)
stop_button2.grid(row=0, column=1, padx=5)
# 버튼 영역
button_frame = tk.Frame(screen2, bg="black")
button_frame.grid(row=2, column=0, columnspan=3, pady=5)


# 하단 옵션 영역
options_frame = tk.Frame(screen2, bg="black")
options_frame.grid(row=3, column=0, columnspan=3, pady=5)  # 옵션 버튼을 위한 Frame (row=3로 변경)

# 옵션 버튼 (None, Sticker, Real)
button_style = {"font": ("Comic Sans MS", 12, "bold"), "bg": "lightgray", "fg": "black", "relief": "flat", "padx": 10, "pady": 5}

none_button = tk.Button(options_frame, text="None",image=capy_mosaic_photo,compound="bottom", **button_style,command=lambda: set_filter_type("none",none_button))
none_button.grid(row=0, column=0, padx=10)

sticker_button = tk.Button(options_frame, text="Sticker",image=capy_sticker_photo,compound="bottom", **button_style,command=lambda: set_filter_type("sticker",sticker_button))
sticker_button.grid(row=0, column=1, padx=10)

real_button = tk.Button(options_frame, text="Real", image=capy_real_photo,compound="bottom",**button_style,command=lambda: set_filter_type("real",real_button))
real_button.grid(row=0, column=2, padx=10)
# 중간 텍스트 영역 (Options와 Entry 사이)
text_frame = tk.Frame(screen2, bg="black")
text_frame.grid(row=4, column=0, columnspan=3, pady=10)  # row=4로 배치

# 텍스트 필드 레이블
text_label = tk.Label(text_frame, text="Predefined Object:", font=("Comic Sans MS", 14), bg="black", fg="white")
text_label.grid(row=0, column=0, sticky="w", padx=10)
# 입력 영역 (Entry + 검색 아이콘 버튼)
entry_frame = tk.Frame(screen2, bg="black")
entry_frame.grid(row=5, column=0, columnspan=3, pady=10)  # row=4로 배치

entry_style = {"font": ("Comic Sans MS", 30), "bg": "lightgray", "fg": "black", "relief": "flat"}
entry_box = tk.Entry(entry_frame, **entry_style, width=20)
entry_box.grid(row=0, column=0, padx=5)

# 검색 버튼 (돋보기)
search_button = tk.Button(entry_frame, image=search_photo, command=save_text_prompt, width = 50)
search_button.grid(row=0, column=1, padx=5)

# Create 버튼 (오른쪽)
create_button = tk.Button(screen2, text="CREATE", **button_style, width=10,command=run_external_script)
create_button.grid(row=4, column=2, padx=20)  # row=4, column=2로 배치

show_frame(screen1)
# 실행
root.mainloop()
