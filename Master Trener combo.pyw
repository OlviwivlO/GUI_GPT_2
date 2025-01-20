import os
from datetime import datetime, timedelta
import shutil
import json
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QProgressBar, QTextEdit, QLabel, QFileDialog, QApplication, QSpinBox, QFormLayout, QCheckBox, QComboBox
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR
import optuna
from PyQt6.QtGui import QIcon

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size, overlap_size=0, file_type='json'):
        self.examples = []
        if file_type == 'json':
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                prompt = item["prompt"]
                response = item["response"]
                full_text = f"{prompt} {response}"
                tokenized_text = tokenizer.encode(full_text, add_special_tokens=True)
                step_size = block_size - overlap_size if overlap_size > 0 else block_size
                for i in range(0, len(tokenized_text), step_size):
                    fragment = tokenized_text[i:i + block_size]
                    if len(fragment) == block_size:
                        self.examples.append(fragment)
        else:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            tokenized_text = tokenizer.encode(text)
            step_size = block_size - overlap_size if overlap_size > 0 else block_size
            for i in range(0, len(tokenized_text) - block_size + 1, step_size):
                self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class TrainingWorker(QThread):
    update_progress = pyqtSignal(int)
    update_log = pyqtSignal(str)
    training_finished = pyqtSignal()

    def __init__(self, model_path, dataset_path, device, batch_size, epochs, gradient_accumulation_steps, overlap_enabled, overlap_size, block_size, file_type, use_gamma):
        super().__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.overlap_enabled = overlap_enabled
        self.overlap_size = overlap_size
        self.block_size = block_size
        self.file_type = file_type
        self.use_gamma = use_gamma
        self.is_running = True

    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.update_log.emit(f"Модель не найдена по указанному пути: {self.model_path}")
                self.training_finished.emit()
                return
            if not os.path.exists(os.path.join(self.model_path, 'tokenizer_config.json')):
                self.update_log.emit(f"Токенизатор не найден по указанному пути: {self.model_path}")
                self.training_finished.emit()
                return

            tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            model.to(self.device)

            overlap_size = self.overlap_size if self.overlap_enabled else 0
            dataset = CustomTextDataset(tokenizer, self.dataset_path, block_size=self.block_size, overlap_size=overlap_size, file_type=self.file_type)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = AdamW(model.parameters(), lr=5e-5)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.95) if self.use_gamma else None

            model.train()
            total_steps = len(data_loader) * self.epochs
            self.start_time = datetime.now()
            accum_steps = 0

            for epoch in range(self.epochs):
                if not self.is_running:
                    break
                for i, batch in enumerate(data_loader):
                    if not self.is_running:
                        break
                    inputs = batch.to(self.device)
                    labels = batch.to(self.device)
                    outputs = model(inputs, labels=labels, attention_mask=inputs != tokenizer.pad_token_id)
                    loss = outputs.loss
                    loss.backward()

                    if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_steps = 0
                    else:
                        accum_steps += 1

                    current_step = epoch * len(data_loader) + i + 1
                    self.update_progress.emit(int((current_step / total_steps) * 100))
                    current_time = datetime.now()
                    elapsed_time = current_time - self.start_time
                    progress = current_step / total_steps
                    remaining_time = timedelta(seconds=(elapsed_time.total_seconds() / progress) - elapsed_time.total_seconds()) if progress > 0 else timedelta(seconds=0)
                    self.update_log.emit(
                        f"Эпоха {epoch + 1}/{self.epochs}, Партия {i + 1}/{len(data_loader)}, "
                        f"Потеря: {loss.item():.10f}, Время: {current_time.strftime('%H:%M:%S')}, "
                        f"Оставшееся время: {str(remaining_time).split('.')[0]}"
                    )
                if scheduler:
                    scheduler.step()
                self.update_log.emit(f"Эпоха {epoch + 1}/{self.epochs} завершена.")

            if self.is_running:
                self.save_model_and_tokenizer(model, tokenizer)
                self.update_log.emit("Обучение завершено успешно.")
            else:
                self.update_log.emit("Обучение остановлено пользователем.")
            self.training_finished.emit()
        except Exception as e:
            self.update_log.emit(f"Произошла ошибка: {str(e)}")
            self.training_finished.emit()

    def stop_training(self):
        self.is_running = False
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            model.to(self.device)
            self.save_model_and_tokenizer(model, tokenizer)
            self.update_log.emit("Обучение было остановлено, модель сохранена.")
        except Exception as e:
            self.update_log.emit(f"Ошибка при сохранении модели или токенизатора: {str(e)}")

    def save_model_and_tokenizer(self, model, tokenizer):
        model_save_path = os.path.join(self.model_path, '')
        tokenizer_save_path = os.path.join(self.model_path, '')
        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path)
        if os.path.exists(tokenizer_save_path):
            shutil.rmtree(tokenizer_save_path)
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(tokenizer_save_path, exist_ok=True)
        try:
            self.update_log.emit(f"Сохранение модели в {model_save_path}")
            model.save_pretrained(model_save_path)
            self.update_log.emit(f"Сохранение токенизатора в {tokenizer_save_path}")
            tokenizer.save_pretrained(tokenizer_save_path)
            self.update_log.emit("Модель и токенизатор успешно сохранены.")
        except Exception as e:
            self.update_log.emit(f"Ошибка при сохранении модели или токенизатора: {str(e)}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()
        self.setWindowTitle('𝐌𝐀𝐒𝐓𝐄𝐑 𝐓𝐑𝐄𝐍𝐄𝐑 𝐆𝐓𝐏 𝐂𝐎𝐌𝐁𝐈𝐍𝐄𝐃')
        self.setGeometry(0, 0, 600, 400)
        self.setWindowIcon(QIcon("C:/Users/Olviw/OneDrive/Документы/GitHub/GUI_GPT_2/ico/Мастер тренеровок.ico"))

    def initUI(self):
        layout = QVBoxLayout()
        self.start_button = QPushButton('📖 Начать обучение 📖', self)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton('🚨 Остановить обучение 🚨', self)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar(self)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.model_label = QLabel('Модель: ❓', self)
        self.dataset_label = QLabel('Датасет: ❓', self)
        self.model_button = QPushButton('📂 Выбрать модель 📂', self)
        self.model_button.clicked.connect(self.select_model)
        self.dataset_button = QPushButton('📄 Выбрать датасет 📄', self)
        self.dataset_button.clicked.connect(self.select_dataset)
        self.file_type_combo = QComboBox(self)
        self.file_type_combo.addItems(['json', 'txt'])
        self.gamma_checkbox = QCheckBox("Использовать gamma", self)
        self.gamma_checkbox.setChecked(True)
        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setRange(1, 102400)
        self.batch_size_input.setValue(8)
        self.epochs_input = QSpinBox(self)
        self.epochs_input.setRange(1, 1000000)
        self.epochs_input.setValue(1)
        self.gradient_accumulation_input = QSpinBox(self)
        self.gradient_accumulation_input.setRange(1, 102400)
        self.gradient_accumulation_input.setValue(8)
        self.block_size_input = QSpinBox(self)
        self.block_size_input.setRange(1, 102400)
        self.block_size_input.setValue(8)
        self.overlap_checkbox = QCheckBox("Включить перекрытие фрагментов", self)
        self.overlap_checkbox.setChecked(True)
        self.overlap_size_input = QSpinBox(self)
        self.overlap_size_input.setRange(1, 102400)
        self.overlap_size_input.setValue(4)
        params_form = QFormLayout()
        params_form.addRow("Тип файла", self.file_type_combo)
        params_form.addRow("Использовать gamma", self.gamma_checkbox)
        params_form.addRow("⏫ Размер партии ⏬", self.batch_size_input)
        params_form.addRow("⏫ Эпохи ⏬", self.epochs_input)
        params_form.addRow("⏫ Шаги накопления градиента ⏬", self.gradient_accumulation_input)
        params_form.addRow("⏫ Размер блока ⏬", self.block_size_input)
        params_form.addRow(self.overlap_checkbox)
        params_form.addRow("⏫ Размер перекрытия ⏬", self.overlap_size_input)
        layout.addWidget(self.model_button)
        layout.addWidget(self.model_label)
        layout.addWidget(self.dataset_button)
        layout.addWidget(self.dataset_label)
        layout.addLayout(params_form)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)
        self.setLayout(layout)

    def select_model(self):
        model_path = QFileDialog.getExistingDirectory(self, "Выбрать директорию модели")
        if model_path:
            self.model_label.setText(f'Модель: {model_path}')
            self.model_path = model_path

    def select_dataset(self):
        file_type = self.file_type_combo.currentText()
        if file_type == 'json':
            dataset_path = QFileDialog.getOpenFileName(self, "Выбрать файл датасета", "", "JSON Files (*.json)")[0]
        else:
            dataset_path = QFileDialog.getOpenFileName(self, "Выбрать файл датасета", "", "Text Files (*.txt);;All Files (*)")[0]
        if dataset_path:
            self.dataset_label.setText(f'Датасет: {dataset_path}')
            self.dataset_path = dataset_path

    def start_training(self):
        self.log_text.clear()
        batch_size = self.batch_size_input.value()
        epochs = self.epochs_input.value()
        gradient_accumulation_steps = self.gradient_accumulation_input.value()
        overlap_enabled = self.overlap_checkbox.isChecked()
        overlap_size = self.overlap_size_input.value()
        block_size = self.block_size_input.value()
        file_type = self.file_type_combo.currentText()
        use_gamma = self.gamma_checkbox.isChecked()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.worker = TrainingWorker(self.model_path, self.dataset_path, device, batch_size, epochs, gradient_accumulation_steps, overlap_enabled, overlap_size, block_size, file_type, use_gamma)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.update_log.connect(self.log_text.append)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_training(self):
        if self.worker:
            self.worker.stop_training()

    def on_training_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append("Обучение завершено.")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
