import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QSlider, QSpinBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
class GPT2ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("")
        self.setGeometry(100, 100, 800, 700)
        self.model = None
        self.tokenizer = None
        self.max_length = 64
        self.min_length = 32
        self.temperature = 0.1
        self.init_ui()
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        title = QLabel("✨ Футуристический чат GPT ✨")
        title.setFont(QFont("Comic Sans MS", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: rgb(0, 200, 255); text-shadow: 1px 1px 5px rgb(0, 100, 150);")
        main_layout.addWidget(title)
        self.chat_display = QTextEdit()
        self.chat_display.setFont(QFont("Arial", 14))
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: rgb(20, 30, 60);
            color: rgb(200, 250, 250);
            border: 2px solid rgb(0, 200, 255);
            border-radius: 10px;
            padding: 10px;
        """)
        main_layout.addWidget(self.chat_display)
        load_button = QPushButton("Загрузить модель")
        load_button.setFont(QFont("Arial", 14))
        load_button.setStyleSheet("background-color: rgb(0, 150, 255); color: white; padding: 10px;")
        load_button.clicked.connect(self.load_model_dialog)
        main_layout.addWidget(load_button)
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Arial", 14))
        self.input_field.setPlaceholderText("Введите текст здесь...")
        self.input_field.setStyleSheet("""
            background-color: rgb(40, 50, 80);
            color: white;
            border: 2px solid rgb(0, 200, 255);
            border-radius: 10px;
            padding: 5px;
        """)
        input_layout.addWidget(self.input_field)
        send_button = QPushButton("Отправить")
        send_button.setFont(QFont("Arial", 14))
        send_button.setStyleSheet("background-color: rgb(0, 200, 150); color: white; padding: 10px;")
        send_button.clicked.connect(self.send_text)
        input_layout.addWidget(send_button)
        main_layout.addLayout(input_layout)
        settings_label = QLabel("Настройки генерации")
        settings_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        settings_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_label.setStyleSheet("color: rgb(200, 250, 250);")
        main_layout.addWidget(settings_label)
        max_length_layout = QHBoxLayout()
        max_length_label = QLabel("Максимальная длина ответа:")
        max_length_label.setFont(QFont("Arial", 14))
        max_length_label.setStyleSheet("color: rgb(200, 250, 250);")
        max_length_layout.addWidget(max_length_label)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setMinimum(1)
        self.max_length_spin.setMaximum(512)
        self.max_length_spin.setValue(self.max_length)
        self.max_length_spin.valueChanged.connect(lambda value: setattr(self, 'max_length', value))
        max_length_layout.addWidget(self.max_length_spin)
        main_layout.addLayout(max_length_layout)
        min_length_layout = QHBoxLayout()
        min_length_label = QLabel("Минимальная длина для полноценного ответа:")
        min_length_label.setFont(QFont("Arial", 14))
        min_length_label.setStyleSheet("color: rgb(200, 250, 250);")
        min_length_layout.addWidget(min_length_label)
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setMinimum(1)
        self.min_length_spin.setMaximum(512)
        self.min_length_spin.setValue(self.min_length)
        self.min_length_spin.valueChanged.connect(lambda value: setattr(self, 'min_length', value))
        min_length_layout.addWidget(self.min_length_spin)
        main_layout.addLayout(min_length_layout)
        temperature_layout = QHBoxLayout()
        temperature_label = QLabel("<<< Точность✧ ✦ ✧Креативность >>>")
        temperature_label.setFont(QFont("Arial", 14))
        temperature_label.setStyleSheet("color: rgb(200, 250, 250);")
        temperature_layout.addWidget(temperature_label)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setMinimum(1)
        self.temperature_slider.setMaximum(10)
        self.temperature_slider.setValue(int(self.temperature * 10))
        self.temperature_slider.valueChanged.connect(
            lambda value: setattr(self, 'temperature', value / 10)
        )
        temperature_layout.addWidget(self.temperature_slider)
        main_layout.addLayout(temperature_layout)
        central_widget.setLayout(main_layout)
    def load_model_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку с моделью")
        if folder_path:
            model_path = Path(folder_path)
            if not model_path.exists():
                self.chat_display.append("<span style='color:red;'>Ошибка: путь к модели не существует.</span>")
                return
            self.model, self.tokenizer = self.load_model(model_path)
            if self.model is not None and self.tokenizer is not None:
                self.chat_display.append("<span style='color:green;'>Привет! О чём поговорим?</span>")
    def load_model(self, model_path):
        try:
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            self.chat_display.append(f"<span style='color:red;'>Ошибка при загрузке модели: {e}</span>")
            return None, None
    def send_text(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            self.chat_display.append("<span style='color:red;'>Ошибка: пустой ввод.</span>")
            return
        if self.model is None or self.tokenizer is None:
            self.chat_display.append("<span style='color:red;'>Ошибка: модель не загружена.</span>")
            return
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            min_length=self.min_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            temperature=self.temperature,
            top_k=30,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.chat_display.append(f"")
        self.chat_display.append(f"{response}")
        self.input_field.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GPT2ChatApp()
    window.show()
    sys.exit(app.exec())
