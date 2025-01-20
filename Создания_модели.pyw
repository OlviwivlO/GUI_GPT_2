import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextEdit,
    QPushButton,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QTabWidget,
    QProgressBar,
    QMessageBox,
    QFileDialog
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


class ModelThread(QThread):
    update_text_signal = pyqtSignal(str)
    model_saved_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            if not os.path.exists(self.config['tokenizer_path']):
                self.update_text_signal.emit(f"Путь к токенизатору не существует: {self.config['tokenizer_path']}")
                return
            
            self.update_text_signal.emit("Загружаем предварительно созданный токенизатор")
            tokenizer = GPT2Tokenizer.from_pretrained(self.config['tokenizer_path'])

            self.update_text_signal.emit("Создаем конфигурацию модели с нужными параметрами")
            model_config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_layer=self.config['n_layer'],
                n_head=self.config['n_head'],
                n_embd=self.config['n_embd'],
                intermediate_size=self.config['intermediate_size'],
                hidden_size=self.config['hidden_size'],
                max_position_embeddings=self.config['max_position_embeddings'],
                num_attention_heads=self.config['num_attention_heads'],
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                sep_token_id=tokenizer.sep_token_id,
                activation_function=self.config['activation_function'],
                initializer_range=self.config['initializer_range'],
                layer_norm_eps=self.config['layer_norm_eps'],
                scale_attn_by_inverse_layer_idx=self.config['scale_attn_by_inverse_layer_idx'],
                reorder_and_upcast_attn=self.config['reorder_and_upcast_attn'],
                use_cache=self.config['use_cache'],
                attention_probs_dropout_prob=self.config['attention_probs_dropout_prob'],
                hidden_dropout_prob=self.config['hidden_dropout_prob']
            )

            self.update_text_signal.emit("Создаем модель на основе заданной конфигурации")
            model = GPT2LMHeadModel(config=model_config)
            model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
            self.update_text_signal.emit("Модель создана.")

            if self.config['gradient_checkpointing']:
                model.gradient_checkpointing_enable()

            # Прогресс сохранения модели
            self.progress_signal.emit(50)
            
            self.update_text_signal.emit("Сохраняем модель и токенизатор.")
            model.save_pretrained(self.config['model_save_path'])
            tokenizer.save_pretrained(self.config['model_save_path'])
            self.update_text_signal.emit("Модель и токенизатор сохранены.")
            
            # Прогресс завершен
            self.progress_signal.emit(100)
            self.model_saved_signal.emit()
        except Exception as e:
            self.update_text_signal.emit(f"Ошибка: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("𝐍𝐄𝐖 𝐌𝐎𝐃𝐄𝐋 𝐆𝐏𝐓")
        self.setGeometry(0, 0, 250, 500)
        self.setWindowIcon(QIcon("C:/Users/Olviw/OneDrive/Документы/GitHub/GUI_GPT_2/ico/Создания модели.ico"))
        
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.createModelTab()
        self.createSettingsTab()

        self.model_thread = None

    def createModelTab(self):
        self.modelTab = QWidget()
        self.tabs.addTab(self.modelTab, "🔄 Создание модели 🔄")

        layout = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        self.exit_button = QPushButton("🌞 Выход 🌞", self)
        self.exit_button.setEnabled(False)
        self.exit_button.clicked.connect(self.close)

        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.exit_button)
        
        self.modelTab.setLayout(layout)

    def createSettingsTab(self):
        self.settingsTab = QWidget()
        self.tabs.addTab(self.settingsTab, "🔧 Настройки 🔨")

        layout = QFormLayout()

        self.tokenizerPathEdit = QLineEdit()
        self.modelSavePathEdit = QLineEdit()
        self.nLayerEdit = QLineEdit("8")                 # Количество слоев
        self.nHeadEdit = QLineEdit("8")                  # Число голов внимания
        self.nEmbdEdit = QLineEdit("256")                # Размерность векторного представления
        self.intermediateSizeEdit = QLineEdit("1024")      # Размер промежуточного слоя
        self.hiddenSizeEdit = QLineEdit("512")            # Размер скрытого слоя
        self.maxPositionEmbeddingsEdit = QLineEdit("1024") # Максимальное количество позиционных эмбеддингов
        self.numAttentionHeadsEdit = QLineEdit("64")      # Количество голов внимания
        self.gradientCheckpointingEdit = QLineEdit("True") # Включение градиентного контрольного сохранения
        self.activationFunctionEdit = QLineEdit("gelu")   # Функция активации
        self.initializerRangeEdit = QLineEdit("0.02")    # Диапазон инициализации
        self.layerNormEpsEdit = QLineEdit("1e-5")        # Эпсилон для нормализации слоя
        self.scaleAttnByInverseLayerIdxEdit = QLineEdit("True") # Масштабирование внимания
        self.reorderAndUpcastAttnEdit = QLineEdit("True") # Переупорядочивание и повышение точности внимания
        self.useCacheEdit = QLineEdit("True")  # Использование кэша
        self.attentionProbsDropoutProbEdit = QLineEdit("0.1")  # Вероятность дропаута внимания
        self.hiddenDropoutProbEdit = QLineEdit("0.1")  # Вероятность дропаута скрытых слоев

        layout.addRow("Путь к токенизатору:", self.tokenizerPathEdit)
        layout.addRow("Путь сохранения модели:", self.modelSavePathEdit)
        
        # Кнопки выбора путей
        self.tokenizerButton = QPushButton("📂 Выбрать токенизатор 📂")
        self.tokenizerButton.clicked.connect(self.choose_tokenizer_path)
        layout.addRow(self.tokenizerButton)

        self.modelSaveButton = QPushButton("📁 Выбрать путь для сохранения модели 📁")
        self.modelSaveButton.clicked.connect(self.choose_model_save_path)
        layout.addRow(self.modelSaveButton)

        layout.addRow("🔼 Количество слоев 🔽", self.nLayerEdit)
        layout.addRow("🔼 Количество голов 🔽", self.nHeadEdit)
        layout.addRow("🔼 Размер эмбеддингов 🔽", self.nEmbdEdit)
        layout.addRow("🔼 Промежуточный размер 🔽", self.intermediateSizeEdit)
        layout.addRow("🔼 Скрытый размер 🔽", self.hiddenSizeEdit)
        layout.addRow("🔼 Макс. позиционные эмбеддинги 🔽", self.maxPositionEmbeddingsEdit)
        layout.addRow("🔼 Количество голов внимания 🔽", self.numAttentionHeadsEdit)
        layout.addRow("💥 Контроль градиентов 💥", self.gradientCheckpointingEdit)
        layout.addRow("💥 Функция активации 💥", self.activationFunctionEdit)
        layout.addRow("💥 Диапазон инициализации 💥", self.initializerRangeEdit)
        layout.addRow("💥 Эпсилон нормализации слоев 💥", self.layerNormEpsEdit)
        layout.addRow("💥 Масштабировать внимание по индексу слоя 💥", self.scaleAttnByInverseLayerIdxEdit)
        layout.addRow("💥 Перестроить и увеличивать внимание 💥", self.reorderAndUpcastAttnEdit)
        layout.addRow("💥 Использование кэша 💥", self.useCacheEdit)
        layout.addRow("💥 Вероятность дропаута внимания 💥", self.attentionProbsDropoutProbEdit)
        layout.addRow("💥 Вероятность дропаута скрытых слоев 💥", self.hiddenDropoutProbEdit)

        self.start_button = QPushButton("🔄 Начать создание модели 🔄")
        self.start_button.clicked.connect(self.start_model_creation)

        layout.addWidget(self.start_button)
        
        self.settingsTab.setLayout(layout)

    def choose_tokenizer_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку с токенизатором")
        if directory:
            self.tokenizerPathEdit.setText(directory)

    def choose_model_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения модели")
        if directory:
            self.modelSavePathEdit.setText(directory)

    def start_model_creation(self):
        try:
            config = {
                'tokenizer_path': self.tokenizerPathEdit.text(),
                'model_save_path': self.modelSavePathEdit.text(),
                'n_layer': int(self.nLayerEdit.text()),
                'n_head': int(self.nHeadEdit.text()),
                'n_embd': int(self.nEmbdEdit.text()),
                'intermediate_size': int(self.intermediateSizeEdit.text()),
                'hidden_size': int(self.hiddenSizeEdit.text()),
                'max_position_embeddings': int(self.maxPositionEmbeddingsEdit.text()),
                'num_attention_heads': int(self.numAttentionHeadsEdit.text()),
                'gradient_checkpointing': self.gradientCheckpointingEdit.text().lower() == 'true',
                'activation_function': self.activationFunctionEdit.text(),
                'initializer_range': float(self.initializerRangeEdit.text()),
                'layer_norm_eps': float(self.layerNormEpsEdit.text()),
                'scale_attn_by_inverse_layer_idx': self.scaleAttnByInverseLayerIdxEdit.text().lower() == 'true',
                'reorder_and_upcast_attn': self.reorderAndUpcastAttnEdit.text().lower() == 'true',
                'use_cache': self.useCacheEdit.text().lower() == 'true',
                'attention_probs_dropout_prob': float(self.attentionProbsDropoutProbEdit.text()),
                'hidden_dropout_prob': float(self.hiddenDropoutProbEdit.text())
            }

            self.model_thread = ModelThread(config)
            self.model_thread.update_text_signal.connect(self.update_text_edit)
            self.model_thread.model_saved_signal.connect(self.model_saved)
            self.model_thread.progress_signal.connect(self.update_progress_bar)
            self.progress_bar.setValue(0)
            self.model_thread.start()
            self.start_button.setEnabled(False)
            self.exit_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def update_text_edit(self, text):
        self.text_edit.append(text)

    def model_saved(self):
        # Теперь кнопка выхода остаётся активной после завершения создания модели
        self.start_button.setEnabled(True)
        # Убедимся, что кнопка "Выход" остаётся активной
        self.exit_button.setEnabled(True)


    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
