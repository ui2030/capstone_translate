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
pytesseract.pytesseract.tesseract_cmd = r"C:\tesseract\tesseract.exe"
import difflib
import re
import os

# Tesseract 경로 설정
tesseract_path = r"C:\tesseract\tesseract.exe"
if os.path.exists(tesseract_path):
    print("Tesseract 경로 설정 완료")
else:
    raise FileNotFoundError("[ERROR] Tesseract 경로가 잘못되었습니다. 경로를 확인해주세요.")

# 번역 모델 설정 (m2m100)
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# 모델을 GPU로 이동 (GPU가 없으면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# PyQt5 고해상도 DPI 스케일링 설정 (QApplication 생성 전)
QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class TransparentWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.old_pos = None

        # 번역된 텍스트 저장 ([(text, bbox, font_size), ...])
        self.translated_text = []

        # 스레드 제어 변수
        self.stop_thread = False
        self.model_loaded = False
        self.setWindowTitle("번역기")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # DPI / 스케일 헬퍼
        self._scale = 1.0
        try:
            self._scale = float(self.devicePixelRatioF())
        except AttributeError:
            try:
                self._scale = float(self.devicePixelRatio())
            except Exception:
                self._scale = 1.0
        def px(v):  # 좌표/크기는 반드시 int
            return int(round(v * self._scale))
        self.px = px  # 인스턴스 함수로 보관

        # UI
        self.initUI()
        self.show()

        # 비동기 OCR/번역 스레드
        self.translation_thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        self.translation_thread.start()
        self.font_size = None

        # 타이핑 효과
        self.typing_index = 0
        self.current_line = ""
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.update_typing)

        # 초기 위치
        self.old_pos = QPoint()

    def initUI(self):
        px = self.px

        # 언어 선택 콤보박스
        self.combo = QComboBox(self)
        self.combo.addItems(["Korean", "French", "German", "Chinese", "English"])
        self.combo.setGeometry(px(10), px(10), px(200), px(30))

        # 모델 로드 버튼
        self.load_button = QPushButton('모델 로드', self)
        self.load_button.setGeometry(px(220), px(10), px(100), px(30))
        self.load_button.clicked.connect(self.load_model)

        # 종료 버튼
        self.close_button = QPushButton('X', self)
        self.close_button.setGeometry(px(340), px(10), px(30), px(30))
        self.close_button.clicked.connect(self.close)

        # 빨간 테두리 영역 (장면 캡처 영역)
        self.red_rect = QRect(px(100), px(50), px(600), px(400))

        # 크기 조절 핸들
        self.size_grip = QSizeGrip(self)
        self.size_grip.setGeometry(self.red_rect.right() - px(10), self.red_rect.bottom() - px(10), px(20), px(20))

    def resizeEvent(self, event):
        px = self.px
        # 윈도우 크기 조절 시 빨간 테두리와 크기 조절 핸들 업데이트
        self.red_rect.setWidth(int(self.width() - px(200)))
        self.red_rect.setHeight(int(self.height() - px(150)))

        # 핸들을 빨간 테두리 우하단에 위치
        self.size_grip.setGeometry(self.red_rect.right() - px(18), self.red_rect.bottom() - px(15), px(24), px(20))
        self.update()

    def run_asyncio_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.capture_and_translate_async())

    # 이미 번역된 텍스트인지 검사
    def is_text_already_translated(self, new_text, bbox):
        for translated_text, translated_bbox, _ in self.translated_text:
            # 텍스트 유사
            if difflib.SequenceMatcher(None, new_text, translated_text).ratio() > 0.9:
                # bbox 유사
                if self.is_bbox_similar(new_text, bbox, [translated_text], [translated_bbox]):
                    return True
        return False

    def is_bbox_similar(self, new_text, new_bbox, existing_texts, existing_bboxes, threshold=10):
        """
        새로운 텍스트/bbox가 기존 것들과 유사한지 비교
        """
        for existing_text, existing_bbox in zip(existing_texts, existing_bboxes):
            text_similarity = difflib.SequenceMatcher(None, new_text, existing_text).ratio()

            x1, y1, x2, y2 = new_bbox
            ex1, ey1, ex2, ey2 = existing_bbox

            bbox_similarity = (
                abs(x1 - ex1) < threshold and
                abs(y1 - ey1) < threshold and
                abs(x2 - ex2) < threshold and
                abs(y2 - ey2) < threshold
            )

            if text_similarity > 0.9 and bbox_similarity:
                return True

        return False

    async def capture_and_translate_async(self):
        while not self.stop_thread:
            if not self.model_loaded:
                await asyncio.sleep(1)
                continue

            try:
                # 빨간 박스 내부 영역 캡처 (이미 device pixels라 나눗셈 불필요)
                x1 = int(self.geometry().x() + self.red_rect.x())
                y1 = int(self.geometry().y() + self.red_rect.y())
                x2 = int(x1 + self.red_rect.width())
                y2 = int(y1 + self.red_rect.height())
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                screenshot_np = np.array(screenshot)

                # OCR 수행
                ocr_results = self.perform_ocr_with_boxes(screenshot_np)

                # 기존 번역된 줄/바운딩박스
                existing_lines = [item[0] for item in self.translated_text]
                existing_bboxes = [item[1] for item in self.translated_text]

                # 새로운 텍스트만 수집
                new_text_boxes = []
                for text, bbox, font_size in ocr_results:
                    if text.strip() and not self.is_bbox_similar(text, bbox, existing_lines, existing_bboxes):
                        new_text_boxes.append((text, bbox, font_size))

                if not new_text_boxes:
                    print("[INFO] 새로운 텍스트가 없습니다. OCR 결과가 반복되지 않도록 처리.")
                    await asyncio.sleep(0.5)
                    continue

                # 비동기 번역
                tasks = [self.async_translate_text(text) for text, _, _ in new_text_boxes]
                translations = await asyncio.gather(*tasks)

                # 번역 결과 저장 (윈도우 내부 상대 bbox → red_rect 기준으로 보정)
                for idx, (text, bbox, font_size) in enumerate(new_text_boxes):
                    adjusted_bbox = (
                        bbox[0] + self.red_rect.x(),
                        bbox[1] + self.red_rect.y(),
                        bbox[2] + self.red_rect.x(),
                        bbox[3] + self.red_rect.y()
                    )
                    self.translated_text.append((translations[idx], adjusted_bbox, font_size))

                # 공백 라인 제거
                self.translated_text = [item for item in self.translated_text if item[0].strip()]

                # 타이핑 인덱스 초기화
                if self.translated_text:
                    self.typing_index = 0
                    self.current_line = ""
                    QMetaObject.invokeMethod(self, "start_typing_timer", Qt.QueuedConnection)

            except Exception as e:
                print(f"[ERROR] {str(e)}")
                self.stop_thread = True
                break

            await asyncio.sleep(0.5)

    def update_typing(self):
        if self.typing_index < len(self.translated_text):
            line, bbox, font_size = self.translated_text[self.typing_index]
            if len(self.current_line) < len(line):
                self.current_line += line[len(self.current_line)]
                QTimer.singleShot(30, self.update)
            else:
                self.typing_index += 1
                self.current_line = ""
                if self.typing_index < len(self.translated_text):
                    QTimer.singleShot(30, self.update)
                else:
                    self.typing_timer.stop()

    def draw_translated_text(self, painter, text, bbox, font_size):
        if isinstance(bbox, tuple) and len(bbox) == 4:
            left, top, right, bottom = bbox
        else:
            print(f"[ERROR] bbox 값이 예상과 다릅니다: {bbox}")
            return

        # 폰트 크기 조정
        adjusted_font_size = max(10, int(round(font_size * self._scale)))
        font = QFont("Arial", adjusted_font_size)

        # 그리기 영역 (red_rect 내부로 클리핑)
        rect = QRect(
            int(max(self.red_rect.left(), left)),
            int(max(self.red_rect.top(), top)),
            int(min(self.red_rect.width(), (right - left))),
            int(min(self.red_rect.height(), (bottom - top)))
        )

        # 배경
        painter.setBrush(QColor(200, 200, 200, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, int(10 * self._scale), int(10 * self._scale))

        # 텍스트
        painter.setPen(QPen(QColor(0, 0, 0), int(2 * self._scale)))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)

    def perform_ocr_with_boxes(self, image):
        try:
            # 전처리: 흑백 + 이진화 + 노이즈 제거
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

            # Tesseract OCR
            print("[INFO] Tesseract OCR 사용 시도 중...")
            tesseract_result = pytesseract.image_to_data(
                cleaned_img, lang="eng+kor", output_type=pytesseract.Output.DICT
            )

            # 디버그: OCR 결과 확인
            for i, text in enumerate(tesseract_result['text']):
                print(f"OCR 텍스트: {text}, bbox: {tesseract_result['left'][i], tesseract_result['top'][i], tesseract_result['width'][i], tesseract_result['height'][i]}")

            ocr_text_boxes = self.process_ocr_result_by_lines(tesseract_result)
            if ocr_text_boxes:
                print("[INFO] Tesseract OCR이 텍스트를 감지했습니다.")
                return ocr_text_boxes
            return []
        except Exception as e:
            print(f"[ERROR] OCR 처리 중 오류 발생: {str(e)}")
            return []

    def process_ocr_result_by_lines(self, ocr_result):
        """ 단어를 줄 단위로 묶어서 반환: (full_text, (l,t,r,b), font_size) 리스트 """
        ocr_text_boxes = []
        current_line = []
        previous_bottom = -1
        line_height_threshold = 1
        line_height_min_threshold = 10

        for i, text in enumerate(ocr_result["text"]):
            if text.strip():
                left = ocr_result["left"][i]
                top = ocr_result["top"][i]
                width = ocr_result["width"][i]
                height = ocr_result["height"][i]
                bottom = top + height

                if previous_bottom != -1:
                    if top - previous_bottom > line_height_threshold:
                        if current_line:
                            ocr_text_boxes.append(self.create_line_entry(current_line))
                        current_line = []
                    current_line.append((text, left, top, width, height))
                else:
                    current_line.append((text, left, top, width, height))

                previous_bottom = bottom

        if current_line:
            ocr_text_boxes.append(self.create_line_entry(current_line))

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

            # 목표 언어
            tgt_lang_code = {
                "Korean": "ko",
                "French": "fr",
                "German": "de",
                "Chinese": "zh",
                "English": "en"
            }[self.combo.currentText()]

            if src_lang_code == tgt_lang_code:
                return text

            # 중간 영어 번역 → 최종 번역
            tokenizer.src_lang = src_lang_code
            inputs = tokenizer(text, return_tensors="pt").to(device)
            intermediate_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
            intermediate_text = tokenizer.decode(intermediate_tokens[0], skip_special_tokens=True)

            tokenizer.src_lang = "en"
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
        pen = QPen(QColor(255, 0, 0), int(5 * self._scale))
        painter.setPen(pen)
        painter.drawRect(self.red_rect)

        if hasattr(self, 'translated_text') and self.translated_text:
            # 완료된 줄들
            for i in range(self.typing_index):
                text, bbox, font_size = self.translated_text[i]
                print(f"출력 중: {text}, bbox: {bbox}")
                self.draw_translated_text(painter, text, bbox, font_size)

            # 현재 타이핑 중인 줄
            if self.typing_index < len(self.translated_text):
                text, bbox, font_size = self.translated_text[self.typing_index]
                self.draw_translated_text(painter, self.current_line, bbox, font_size)

    def closeEvent(self, event):
        self.stop_thread = True
        if hasattr(self, "translation_thread") and self.translation_thread.is_alive():
            self.translation_thread.join()
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