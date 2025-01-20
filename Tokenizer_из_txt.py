import sys
import os
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QFormLayout, QStatusBar, QSpinBox, QLineEdit
)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class TokenizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("𝐓𝐎𝐊𝐄𝐍𝐈𝐙𝐄𝐑")
        self.setGeometry(0, 0, 800, 600)
        self.setWindowIcon(QIcon("C:/Users/Olviw/OneDrive/Документы/GitHub/GUI_GPT_2/ico/Токенизатор zero.ico"))

        self.default_tokens = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Font setup
        font = QFont("Segoe UI", 12)
        self.setFont(font)

        # Token file selection
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setPlaceholderText("Выберите текстовый файл с данными для обучения токенизатора")
        self.file_line_edit.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.file_line_edit.setFont(QFont("Segoe UI", 14))

        self.file_button = QPushButton("📂 Выбрать файл 📂")
        self.file_button.setStyleSheet("background-color: #808080; color: #FFFFFF; border: none; padding: 10px; border-radius: 5px;")
        self.file_button.setFont(QFont("Segoe UI", 14))
        self.file_button.clicked.connect(self.select_text_file)

        # Vocabulary size
        self.vocab_size_input = QSpinBox()
        self.vocab_size_input.setMinimum(1)
        self.vocab_size_input.setMaximum(1000000000)
        self.vocab_size_input.setValue(1)
        self.vocab_size_input.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.vocab_size_input.setFont(QFont("Segoe UI", 14))

        # Maximum length
        self.max_len_input = QSpinBox()
        self.max_len_input.setMinimum(1)
        self.max_len_input.setMaximum(1000000000)
        self.max_len_input.setValue(1)
        self.max_len_input.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.max_len_input.setFont(QFont("Segoe UI", 14))

        # Output directory selection
        self.output_dir_line_edit = QLineEdit()
        self.output_dir_line_edit.setPlaceholderText("Выберите папку для сохранения")
        self.output_dir_line_edit.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.output_dir_line_edit.setFont(QFont("Segoe UI", 14))

        self.output_dir_button = QPushButton("📂 Выбрать папку 📂")
        self.output_dir_button.setStyleSheet("background-color: #808080; color: #FFFFFF; border: none; padding: 10px; border-radius: 5px;")
        self.output_dir_button.setFont(QFont("Segoe UI", 14))
        self.output_dir_button.clicked.connect(self.select_output_dir)

        # Start button
        self.start_button = QPushButton("✅ Создать токенизатор ✅")
        self.start_button.setStyleSheet("background-color: #006400; color: #FFFFFF; border: none; padding: 10px; border-radius: 5px;")
        self.start_button.setFont(QFont("Segoe UI", 14))
        self.start_button.clicked.connect(self.create_tokenizer)

        # Layout
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("📁 Текстовый файл 📁"), self.file_line_edit)
        form_layout.addRow(self.file_button)
        form_layout.addRow(QLabel("📑 Размер словаря (vocab_size) 📑"), self.vocab_size_input)
        form_layout.addRow(QLabel("📏 Максимальная длина (max_len) 📏"), self.max_len_input)
        form_layout.addRow(QLabel("📁 Папка для сохранения 📁"), self.output_dir_line_edit)
        form_layout.addRow(self.output_dir_button)

        layout.addLayout(form_layout)
        layout.addWidget(self.start_button)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("background-color: #333; color: #FFFFFF; font: 12pt 'Segoe UI';")

    def select_text_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите текстовый файл", filter="Text files (*.txt)")
        if file_path:
            self.file_line_edit.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if dir_path:
            self.output_dir_line_edit.setText(dir_path)

    def create_tokenizer(self):
        text_file = self.file_line_edit.text()
        output_dir = self.output_dir_line_edit.text()

        if not text_file:
            self.status_bar.showMessage("Необходимо выбрать текстовый файл.")
            return

        if not output_dir:
            self.status_bar.showMessage("Необходимо выбрать папку для сохранения.")
            return

        vocab_size = self.vocab_size_input.value()
        max_len = self.max_len_input.value()

        self.status_bar.showMessage("Создание токенизатора. Пожалуйста, подождите...")

        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                data = f.readlines()

            tokens = self.default_tokens
            self.generate_tokenizer(output_dir, tokens, vocab_size, max_len, data)
            self.status_bar.showMessage("Токенизатор успешно создан!")
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка: {str(e)}")

    def generate_tokenizer(self, output_dir, tokens, vocab_size, max_len, data):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=tokens,
            show_progress=True
        )

        tokenizer.train_from_iterator(data, trainer)

        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        vocab = tokenizer.get_vocab()
        with open(os.path.join(output_dir, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            merges = tokenizer_data['model']['merges']
            with open(os.path.join(output_dir, "merges.txt"), 'w', encoding='utf-8') as merges_file:
                for merge in merges:
                    merges_file.write(f"{merge}\n")

        special_tokens = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        with open(os.path.join(output_dir, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "max_len": max_len,
                "do_lower_case": False
            }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TokenizerApp()
    window.show()
    sys.exit(app.exec())
