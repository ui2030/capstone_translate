import sys
import numpy as np
import asyncio
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QMetaObject, Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QPushButton, QSizeGrip
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QRect, QTimer, QPoint
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import threading
from langdetect import detect
from PIL import Image
from PIL import ImageGrab
import pytesseract
import difflib
import re
import os

# Tesseract 경로 설정
tesseract_path = r"C:/Users/ui2030/box/tesseract/tesseract.exe"
if os.path.exists(tesseract_path):
    print("Tesseract 경로 설정 완료")
else:
    raise FileNotFoundError("[ERROR] Tesseract 경로가 잘못되었습니다. 경로를 확인해주세요.")

# 번역 모델 설정 (m2m100)
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# model = M2M100ForConditionalGeneration.from_pretrained('./fine_tuned_m2m100')
# tokenizer = M2M100Tokenizer.from_pretrained('./fine_tuned_m2m100')

# 모델을 GPU로 이동 (GPU가 없으면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# PyQt5 고해상도 DPI 스케일링 설정
QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class TransparentWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.old_pos = None  # self.old_pos를 초기화

        # 번역된 텍스트를 저장하는 리스트 초기화
        self.translated_text = []  # 번역된 텍스트를 저장할 리스트

        # 스레드 제어 변수
        self.stop_thread = False
        self.model_loaded = False
        self.setWindowTitle("번역기")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # DPI 설정
        self.device_pixel_ratio = QApplication.primaryScreen().devicePixelRatio()

        # UI 설정
        self.initUI()
        self.show()

        # 비동기 OCR 및 번역 스레드 실행 (UI 초기화 후)
        self.translation_thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        self.translation_thread.start()
        self.font_size = None

        # 타이핑 효과 관련 변수
        self.typing_index = 0
        self.current_line = ""
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.update_typing)

        # 초기 위치 정의 (AttributeError 방지)
        self.old_pos = QPoint()

    def initUI(self):
        # 언어 선택 콤보박스
        self.combo = QComboBox(self)
        self.combo.addItems(["Korean", "French", "German", "Chinese", "English"])
        self.combo.setGeometry(10 * self.device_pixel_ratio, 10 * self.device_pixel_ratio,
                               200 * self.device_pixel_ratio, 30 * self.device_pixel_ratio)

        # 모델 로드 버튼
        self.load_button = QPushButton('모델 로드', self)
        self.load_button.setGeometry(220 * self.device_pixel_ratio, 10 * self.device_pixel_ratio,
                                     100 * self.device_pixel_ratio, 30 * self.device_pixel_ratio)
        self.load_button.clicked.connect(self.load_model)

        # 종료 버튼
        self.close_button = QPushButton('X', self)
        self.close_button.setGeometry(340 * self.device_pixel_ratio, 10 * self.device_pixel_ratio,
                                      30 * self.device_pixel_ratio, 30 * self.device_pixel_ratio)
        self.close_button.clicked.connect(self.close)

        # 빨간 테두리 영역
        self.red_rect = QRect(100 * self.device_pixel_ratio, 50 * self.device_pixel_ratio,
                              600 * self.device_pixel_ratio, 400 * self.device_pixel_ratio)

        # 크기 조절 핸들 추가
        self.size_grip = QSizeGrip(self)
        self.size_grip.setGeometry(self.red_rect.right() - 10, self.red_rect.bottom() - 10, 20, 20)

    def resizeEvent(self, event):
        # 윈도우 크기 조절 시 빨간 테두리와 크기 조절 핸들 업데이트
        self.red_rect.setWidth(self.width() - 200 * self.device_pixel_ratio)
        self.red_rect.setHeight(self.height() - 150 * self.device_pixel_ratio)

        # 크기 조절 핸들을 빨간 테두리 오른쪽 아래에 붙여서 위치 조정
        self.size_grip.setGeometry(self.red_rect.right() - 17.5, self.red_rect.bottom() - 15, 24, 20)
        self.update()

    def run_asyncio_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.capture_and_translate_async())

        #이미 번역된 텍스트를 처리하는 함수
    def is_text_already_translated(self, new_text, bbox):
        # 번역된 텍스트가 존재하는지 확인
        for translated_text, translated_bbox, _ in self.translated_text:
        # 유사한 텍스트가 이미 번역된 경우
            if difflib.SequenceMatcher(None, new_text, translated_text).ratio() > 0.9:
        # bbox가 비슷한 경우(같은 텍스트일 가능성 있음)
                if self.is_bbox_similar(bbox, translated_bbox):
                    return True
        return False

    def is_bbox_similar(self, new_text, new_bbox, existing_texts, existing_bboxes, threshold=10):
        """ 
        새로운 텍스트와 bbox가 기존 텍스트 및 bbox와 유사한지 비교
        텍스트 유사도 및 bbox 유사도 모두 고려.
        """
        for existing_text, existing_bbox in zip(existing_texts, existing_bboxes):
            # 텍스트 유사도 비교
            text_similarity = difflib.SequenceMatcher(None, new_text, existing_text).ratio()
            
            # bbox 유사도 비교 (bbox 위치와 크기를 비교)
            x1, y1, x2, y2 = new_bbox
            ex1, ey1, ex2, ey2 = existing_bbox
            
            # bbox 유사도를 비교: 비율 차이가 일정 범위 내에 있으면 유사하다고 판단
            bbox_similarity = (
                abs(x1 - ex1) < threshold and
                abs(y1 - ey1) < threshold and
                abs(x2 - ex2) < threshold and
                abs(y2 - ey2) < threshold
            )
            
            # 텍스트 유사도와 bbox 유사도를 결합하여 판단
            if text_similarity > 0.9 and bbox_similarity:  # threshold를 0.9로 설정
                return True
        
        return False

    async def capture_and_translate_async(self):
        while not self.stop_thread:
            if not self.model_loaded:
                await asyncio.sleep(1)
                continue

            try:
                # 빨간색 박스 내부 영역 캡처
                x1 = int(self.geometry().x() + self.red_rect.x() / self.device_pixel_ratio)
                y1 = int(self.geometry().y() + self.red_rect.y() / self.device_pixel_ratio)
                x2 = int(x1 + self.red_rect.width() / self.device_pixel_ratio)
                y2 = int(y1 + self.red_rect.height() / self.device_pixel_ratio)
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                screenshot_np = np.array(screenshot)

                # OCR 수행
                ocr_results = self.perform_ocr_with_boxes(screenshot_np)

                # 기존 번역된 텍스트에 있는 줄 수집
                existing_lines = [item[0] for item in self.translated_text]
                existing_bboxes = [item[1] for item in self.translated_text]

                # 유사도를 기준으로 새로운 텍스트만 수집
                new_text_boxes = []
                for text, bbox, font_size in ocr_results:  # 수정된 ocr_results 사용
                    if text.strip() and not self.is_bbox_similar(text, bbox, existing_lines, existing_bboxes):
                        new_text_boxes.append((text, bbox, font_size))

                if not new_text_boxes:
                    print("[INFO] 새로운 텍스트가 없습니다. OCR 결과가 반복되지 않도록 처리.")
                    await asyncio.sleep(0.5)
                    continue

                # 비동기 번역 작업 수행
                tasks = [self.async_translate_text(text) for text, _, _ in new_text_boxes]
                translations = await asyncio.gather(*tasks)

                # 번역된 결과 저장 (기존 번역된 텍스트에 추가)
                for idx, (text, bbox, font_size) in enumerate(new_text_boxes):
                    adjusted_bbox = (
                        bbox[0] + self.red_rect.x(),
                        bbox[1] + self.red_rect.y(),
                        bbox[2] + self.red_rect.x(),
                        bbox[3] + self.red_rect.y()
                    )
                    self.translated_text.append((translations[idx], adjusted_bbox, font_size))

                # 번역된 텍스트를 최신 상태로 반영
                self.translated_text = [item for item in self.translated_text if item[0].strip()]

                # 타이핑이 끝났으면 다음 줄로 넘어가도록 처리
                if self.typing_index == len(self.translated_text): 
                    self.typing_index = 0
                    self.current_line = ""

                # 번역 결과를 화면에 반영
                if self.translated_text:
                    self.typing_index = 0  # 번역된 텍스트 인덱스 초기화
                    self.current_line = ""  # 새로운 줄로 넘어가기
                    QMetaObject.invokeMethod(self, "start_typing_timer", Qt.QueuedConnection)

            except Exception as e:
                print(f"[ERROR] {str(e)}")
                self.stop_thread = True  # 예외 발생 시 루프 종료
                break  # 루프를 빠져나가도록 수정

            await asyncio.sleep(0.5)  # 너무 빠르게 반복되지 않도록 대기

    def update_typing(self):
        # 타이핑 애니메이션 구현
        if self.typing_index < len(self.translated_text):
            line, bbox, font_size = self.translated_text[self.typing_index]
            if len(self.current_line) < len(line):  # 아직 타이핑이 진행 중이면
                self.current_line += line[len(self.current_line)]  # 한 글자씩 추가
                QTimer.singleShot(30, self.update)  # 30ms 후에 다시 호출하여 타이핑 계속
            else:
                # 한 줄이 끝나면 다음 줄로 넘어감
                self.typing_index += 1
                self.current_line = ""  # 다음 줄을 위해 current_line 초기화
                if self.typing_index < len(self.translated_text):
                    QTimer.singleShot(30, self.update)  # 다음 줄로 넘어가면서 타이핑 시작
                else:
                    self.typing_timer.stop()  # 모든 텍스트가 타이핑되었으면 타이머를 멈춤

    def draw_translated_text(self, painter, text, bbox, font_size):
        # bbox의 값이 올바른지 확인하고, 글자 크기를 적절히 조정
        if isinstance(bbox, tuple) and len(bbox) == 4:
            left, top, right, bottom = bbox
        else:
            print(f"[ERROR] bbox 값이 예상과 다릅니다: {bbox}")
            return

        # 글자 크기와 DPI가 적절히 조정되었는지 확인
        adjusted_font_size = max(10, font_size * self.device_pixel_ratio)  # 최소 크기를 10으로 설정
        font = QFont("Arial", int(adjusted_font_size))  # 글자 크기 설정

        rect = QRect(
            int(max(self.red_rect.left(), left * self.device_pixel_ratio)),
            int(max(self.red_rect.top(), top * self.device_pixel_ratio)),
            int(min(self.red_rect.width(), (right - left) * self.device_pixel_ratio)),
            int(min(self.red_rect.height(), (bottom - top) * self.device_pixel_ratio))
        )

        # 텍스트 배경색 설정
        painter.setBrush(QColor(200, 200, 200, 200))  # 투명한 배경
        painter.setPen(Qt.NoPen)  # 테두리 없앰
        painter.drawRoundedRect(rect, int(10 * self.device_pixel_ratio), int(10 * self.device_pixel_ratio))

        # 글자 색과 테두리 설정
        painter.setPen(QPen(QColor(0, 0, 0), int(2 * self.device_pixel_ratio)))  # 글자 색, 테두리
        painter.setFont(font)  # 글자 크기 설정
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)  # 텍스트 출력

    def perform_ocr_with_boxes(self, image):
        try:
            # 이미지 전처리: 흑백 변환 + 이진화 + 노이즈 제거
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
            _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이진화 처리
            kernel = np.ones((1, 1), np.uint8)
            cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)  # 노이즈 제거

            # Tesseract OCR 사용 시도
            print("[INFO] Tesseract OCR 사용 시도 중...")
            tesseract_result = pytesseract.image_to_data(
                cleaned_img, lang="eng+kor", output_type=pytesseract.Output.DICT
            )

            # OCR 결과에서 텍스트와 bbox를 출력하여 점검
            for i, text in enumerate(tesseract_result['text']):
                print(f"OCR 텍스트: {text}, bbox: {tesseract_result['left'][i], tesseract_result['top'][i], tesseract_result['width'][i], tesseract_result['height'][i]}")

            ocr_text_boxes = self.process_ocr_result_by_lines(tesseract_result)
            if ocr_text_boxes:
                print("[INFO] Tesseract OCR이 텍스트를 감지했습니다.")
                return ocr_text_boxes
        except Exception as e:
            print(f"[ERROR] OCR 처리 중 오류 발생: {str(e)}")
            return []

    def process_ocr_result_by_lines(self, ocr_result):
        """ Tesseract OCR 결과에서 단어를 줄 단위로 묶어서 반환 """
        ocr_text_boxes = []
        current_line = []
        previous_bottom = -1  # 이전 줄의 하단 위치
        line_height_threshold = 1  # 줄 사이의 최대 허용 간격 (이 값을 통해 줄 구분)
        line_height_min_threshold = 10  # 글자 높이를 최소 기준으로 간격 설정

        # Tesseract 결과에서 각 단어를 순차적으로 처리
        for i, text in enumerate(ocr_result["text"]):
            if text.strip():  # 비어 있지 않은 텍스트만 처리
                left, top, width, height = ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]
                bottom = top + height  # 단어의 하단 위치

                # 줄 구분 기준: 직전 줄과의 간격이 threshold 이상이면 새로운 줄로 구분
                if previous_bottom != -1:
                    # 간격이 큰 경우에는 새로운 줄로 처리
                    if top - previous_bottom > line_height_threshold:
                        if current_line:
                            ocr_text_boxes.append(self.create_line_entry(current_line))
                        current_line = []  # 새로운 줄 시작

                    # 작은 글자와 큰 글자 간에 구분을 두기 위한 최소 높이 기준을 추가
                    elif height > line_height_min_threshold:
                        current_line.append((text, left, top, width, height))  # 큰 글자 처리
                    else:
                        current_line.append((text, left, top, width, height))  # 작은 글자도 처리

                else:
                    # 첫 번째 단어는 항상 current_line에 추가
                    current_line.append((text, left, top, width, height))

                previous_bottom = bottom  # 현재 단어의 하단 위치 저장

        # 마지막 줄 처리
        if current_line:
            ocr_text_boxes.append(self.create_line_entry(current_line))  # 마지막 줄 추가

        return ocr_text_boxes

    def create_line_entry(self, line_words):
        full_text = " ".join([word[0] for word in line_words])
        left = min(word[1] for word in line_words)
        top = min(word[2] for word in line_words)
        right = max(word[1] + word[3] for word in line_words)
        bottom = max(word[2] + word[4] for word in line_words)
        font_size = max(word[4] for word in line_words)
        bbox = (left, top, right, bottom)

        return (full_text, bbox, font_size)

    async def async_translate_text(self, text):
        try:
            # 입력 언어 검지
            src_lang_detected = detect(text)
            src_lang_code = src_lang_detected if src_lang_detected in ["ko", "fr", "de", "zh", "en"] else "en"
            
            # 목표 언어 설정
            tgt_lang_code = {
                "Korean": "ko",
                "French": "fr",
                "German": "de",
                "Chinese": "zh",
                "English": "en"
            }[self.combo.currentText()]  # 사용자 선택에 따른 목표 언어 설정

            if src_lang_code == tgt_lang_code:
                return text  # 입력 언어와 목표 언어가 같으면 번역하지 않음

            # 입력 언어 설정
            tokenizer.src_lang = src_lang_code
            inputs = tokenizer(text, return_tensors="pt").to(device)
            intermediate_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
            intermediate_text = tokenizer.decode(intermediate_tokens[0], skip_special_tokens=True)

            # 영어로 번역된 중간 텍스트를 최종 목표 언어로 번역
            tokenizer.src_lang = "en"  # 영어로 중간 번역
            inputs = tokenizer(intermediate_text, return_tensors="pt").to(device)
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang_code))

            return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def load_model(self):
        try:
            self.model_loaded = True
            print("[INFO] 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"[ERROR] 모델 로드 중 오류 발생: {e}")

    @pyqtSlot()
    def start_typing_timer(self):
        self.typing_timer.start(50)

        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 0, 0), 5 * self.device_pixel_ratio)
        painter.setPen(pen)
        painter.drawRect(self.red_rect)

        if hasattr(self, 'translated_text') and self.translated_text:
            # 각 줄을 출력
            for i in range(self.typing_index):
                text, bbox, font_size = self.translated_text[i]
                print(f"출력 중: {text}, bbox: {bbox}")  # 디버그용 로그 추가
                self.draw_translated_text(painter, text, bbox, font_size)
            
            # 현재 타이핑 중인 줄은 진행 중인 부분만 출력
            if self.typing_index < len(self.translated_text):
                text, bbox, font_size = self.translated_text[self.typing_index]
                self.draw_translated_text(painter, self.current_line, bbox, font_size)

    def closeEvent(self, event):
        # 프로그램 종료 시 스레드를 종료할 수 있도록 설정
        self.stop_thread = True
        self.translation_thread.join()  # 스레드 작업이 완료될 때까지 기다림
        event.accept()


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentWindow()
    sys.exit(app.exec_())