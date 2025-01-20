import sys
import os
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit, QLabel, QFormLayout, QStatusBar, QLineEdit, QSpinBox
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

        # Расширенный список специальных токенов
        self.default_tokens = [
            '<unk>', '<s>', '</s>', '<pad>', '<mask>',   # Базовые токены
            '<sep>', '<cls>', '<user>', '<assistant>',   # Токены для диалогов
            '<action>', '<content>', '<summary>',        # Токены для действий и содержания
            '<url>', '<web>',                            # Токены для работы с веб-данными

            # Пробелы и пунктуация
            ' ', ',', '.', '!', '—', '-', ';', ':', '(', ')', '"', '?', '«', '»', "'", '[', ']', '…',
            '{', '}', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '~', '`', '<', '>', '„',
    
            # Кириллица (маленькие буквы)
            'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п',
            'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
    
            # Кириллица (большие буквы)
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П',
            'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
    
            # Латиница (маленькие буквы)
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    
            # Латиница (большие буквы)
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    
            # Цифры
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

            # Математические и логические символы
            '≠', '≈', '±', '√', '∞', '∫', '∑', '∏', '∂', '∇', '→', '←', '⇔', '∀', '∃', '∧', '∨', '⊥', '⊂', '⊃', 'Θ',
    
            # Символы валют
            '₽', '$', '€', '£', '¥', '₴', '₹', '₩', '฿', '₿', '¢', '৳',
            
            # Прочие символы
            '°', '©', '®', '™', '§', '¶',

            #  Диакритические знаки (латинские буквы с акцентами)
            'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ä', 'ö', 'ß', 'ç'          
        ]

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Установка шрифта
        font = QFont("Segoe UI", 12)
        self.setFont(font)

        # Input for tokens
        self.tokens_text_edit = QTextEdit()
        self.tokens_text_edit.setPlaceholderText("Введите токены через запятую. По умолчанию: <unk>, <s>, </s>, <pad>, <mask>, <sep>, <cls>, <user>, <assistant>, <action>, <content>, <summary>, <url>, <web>")
        self.tokens_text_edit.setText(', '.join(self.default_tokens))  # Установка токенов по умолчанию
        self.tokens_text_edit.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.tokens_text_edit.setFont(QFont("Segoe UI", 14))

        # Поле для ввода vocab_size
        self.vocab_size_input = QSpinBox()
        self.vocab_size_input.setMinimum(1)
        self.vocab_size_input.setMaximum(1024000000)
        self.vocab_size_input.setValue(512)  # Задано значение по умолчанию
        self.vocab_size_input.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #444;")
        self.vocab_size_input.setFont(QFont("Segoe UI", 14))

        # Поле для ввода max_len
        self.max_len_input = QSpinBox()
        self.max_len_input.setMinimum(1)
        self.max_len_input.setMaximum(1024000000)
        self.max_len_input.setValue(2048)  # Задано значение по умолчанию
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

        # Layout setup
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("📖 Токены 📖"), self.tokens_text_edit)
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

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if dir_path:
            self.output_dir_line_edit.setText(dir_path)

    def create_tokenizer(self):
        output_dir = self.output_dir_line_edit.text()
        tokens_text = self.tokens_text_edit.toPlainText()

        if not output_dir:
            self.status_bar.showMessage("Необходимо выбрать папку для сохранения.")
            return

        if not tokens_text:
            self.status_bar.showMessage("Необходимо ввести токены.")
            return

        vocab_size = self.vocab_size_input.value()
        max_len = self.max_len_input.value()

        self.status_bar.showMessage("Создание токенизатора. Пожалуйста, подождите...")

        try:
            tokens = [token.strip() for token in tokens_text.split(',') if token.strip()]
            self.generate_tokenizer(output_dir, tokens, vocab_size, max_len)
            self.status_bar.showMessage("Токенизатор успешно создан!")
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка: {str(e)}")

    def generate_tokenizer(self, output_dir, tokens, vocab_size, max_len):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=tokens,
            show_progress=True
        )

        data = self.tokens_text_edit.toPlainText().splitlines()
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

        # Добавляем специальные токены в файл special_tokens_map.json
        special_tokens = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "user_token": "<user>",
            "assistant_token": "<assistant>",
            "action_token": "<action>",
            "content_token": "<content>",
            "summary_token": "<summary>",
            "url_token": "<url>",
            "web_token": "<web>"
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
