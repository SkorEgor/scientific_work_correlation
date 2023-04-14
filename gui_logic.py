from data_and_processing import DataAndProcessing
from gui import Ui_Dialog

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib

matplotlib.use("Qt5Agg")


# Входной список, парсит по столбцам
def parser_all_data(string_list, frequency_list, gamma_list):
    frequency_list.clear()
    gamma_list.clear()

    skipping_first_line = True
    for line in string_list:
        if skipping_first_line:
            skipping_first_line = False
            continue
        if line[0] == "*":
            break

        row = line.split()
        frequency_list.append(float(row[1]))
        gamma_list.append(float(row[4]))


# Входной список, парсит по столбцам, в заданных частотах
def parser(string_list, frequency_list, gamma_list, start_frequency, end_frequency):
    frequency_list.clear()
    gamma_list.clear()

    skipping_first_line = True
    for line in string_list:
        # Пропуск первой строки
        if skipping_first_line:
            skipping_first_line = False
            continue

        # Если звездочки, конец файла
        if line[0] == "*":
            break

        row = line.split()

        # Если частота в диапазоне частот берем
        if start_frequency < float(row[1]) < end_frequency:
            frequency_list.append(float(row[1]))
            gamma_list.append(float(row[4]))


# Проверяет значение на число и положительность (проценты)
def frequency_input_check(frequency, field_name):
    try:
        frequency = float(frequency)

    except ValueError:
        QMessageBox.about(None, "Ошибка ввода", "Введите число в поле '" + field_name + "'.")
        return False

    # Проверка положительности
    if frequency < 0:
        QMessageBox.about(None, "Ошибка ввода", "Введите положительное число в поле '" + field_name + "'.")
        return False

    return True


# Проверяет значение на число и диапазон от 0 до 100 (проценты)
def percentage_check(percentage):
    try:
        percentage = float(percentage)

    except ValueError:
        QMessageBox.about(None, "Ошибка ввода", "Введите число в поле порога.")
        return False

    # Проверка
    if percentage < 0 or percentage > 100:
        QMessageBox.about(None, "Ошибка ввода", "Пороговое значение должно быть от 0 до 100.")
        return False

    return True


class GuiProgram(Ui_Dialog):

    def __init__(self, dialog):
        # Метод создает окно
        Ui_Dialog.__init__(self)  # Initialize Window

        # Объект данных и обработки их
        self.data_signals = DataAndProcessing()

        # Параметры 1 графика
        self.ax1 = None
        self.fig1 = None
        self.canvas1 = None
        self.toolbar1 = None

        # Параметры 2 графика
        self.ax2 = None
        self.fig2 = None
        self.canvas2 = None
        self.toolbar2 = None

        self.setupUi(dialog)  # Set up the UI

        # Initialize the figure in our window
        figure1 = Figure()  # Prep empty figure
        axis1 = figure1.add_subplot(111)  # Prep empty plot
        self.initialize_figure(figure1, axis1)  # Initialize!

        # Initialize the figure in our window
        figure2 = Figure()  # Prep empty figure
        axis2 = figure2.add_subplot(111)  # Prep empty plot
        self.initialize_figure2(figure2, axis2)  # Initialize!

        # Подключаем обработчики нажатий
        self.pushButton_loading_empty_data.clicked.connect(self.plotting_without_noise)
        self.pushButton_loading_signal_data.clicked.connect(self.signal_plotting)
        self.pushButton_signal_difference.clicked.connect(self.signal_difference)
        self.pushButton_update_threshold.clicked.connect(self.threshold)
        self.pushButton_save_table_to_file.clicked.connect(self.saving_data)

        self.tableWidget_frequency_absorption.cellClicked.connect(self.get_clicked_cell)
        self.checkBox_read_filter.toggled.connect(self.filter_state_changed)

        self.lineEdit_filter_frequency_start.textChanged.connect(self.filter_changed)
        self.lineEdit_filter_frequency_end.textChanged.connect(self.filter_changed)

        self.initialize_table()

        # Задаем начало сценария, активные и не активные кнопки
        self.state1_initial()

    # Сценарий: Изменился фильтр чтения файлов
    def filter_changed(self):
        self.pushButton_loading_signal_data.setEnabled(False)

    # Сценарий: Начальное состояние
    def state1_initial(self):
        self.pushButton_loading_empty_data.setEnabled(True)
        self.pushButton_loading_signal_data.setEnabled(False)
        self.pushButton_signal_difference.setEnabled(False)
        self.pushButton_update_threshold.setEnabled(False)
        self.pushButton_save_table_to_file.setEnabled(False)

    # Сценарий: Загружен сигнал без шума
    def state2_loaded_empty(self):
        self.pushButton_loading_empty_data.setEnabled(True)
        self.pushButton_loading_signal_data.setEnabled(True)
        self.pushButton_signal_difference.setEnabled(False)
        self.pushButton_update_threshold.setEnabled(False)
        self.pushButton_save_table_to_file.setEnabled(False)

        self.ax2.clear()
        self.canvas2.draw()

        self.tableWidget_frequency_absorption.setRowCount(0)

    # Сценарий: Загружены оба графика
    def state3_data_loaded(self):
        self.pushButton_loading_empty_data.setEnabled(True)
        self.pushButton_loading_signal_data.setEnabled(False)
        self.pushButton_signal_difference.setEnabled(True)
        self.pushButton_update_threshold.setEnabled(False)
        self.pushButton_save_table_to_file.setEnabled(False)

    # Сценарий: Есть разница сигналов
    def state4_difference(self):
        self.pushButton_loading_empty_data.setEnabled(True)
        self.pushButton_loading_signal_data.setEnabled(False)
        self.pushButton_signal_difference.setEnabled(True)
        self.pushButton_update_threshold.setEnabled(True)
        self.pushButton_save_table_to_file.setEnabled(True)

    # Сценарий: Изменился checkbox вкл/выкл фильтра чтения
    def filter_state_changed(self):
        state = self.checkBox_read_filter.checkState()
        self.lineEdit_filter_frequency_start.setEnabled(state)
        self.lineEdit_filter_frequency_end.setEnabled(state)

    # Кнопка сохранения таблицы
    def saving_data(self):
        # Проверка, что данные для сохранения есть
        if self.data_signals.frequency_peak == [] or self.data_signals.gamma_peak == []:
            QMessageBox.about(None, "Ошибка данных", "Нет данных для сохранения.")
            return

        # Рек-мое название файла
        recommended_file_name = "F" + str(self.data_signals.signal_frequency[0]) + "-" + str(
            self.data_signals.signal_frequency[
                len(self.data_signals.signal_frequency) - 1]) + "_threshold-" + self.lineEdit_threshold.text()

        # Окно с выбором места сохранения
        file_name, file_type = QFileDialog.getSaveFileName(
            None,
            'Сохранение',
            recommended_file_name,
            "Text(*.txt);;Spectrometer Data(*.csv);;All Files(*)"
        )

        # Если имя не получено, прервать
        if not file_name:
            return

        # Открываем файл для чтения
        with open(file_name, "w") as file:

            # Заголовок/Название столбцов
            file.write("FREQUENCY:\tGAMMA:\n")

            # Перебираем по парно частоты и гаммы пиков; Записываем по строчно в файл
            for i in range(self.tableWidget_frequency_absorption.rowCount()):
                if self.tableWidget_frequency_absorption.cellWidget(i, 2).checkState() == Qt.CheckState.Checked:
                    f = self.tableWidget_frequency_absorption.item(i, 0).text()
                    g = self.tableWidget_frequency_absorption.item(i, 1).text()
                    file.write(f + "\t" + g + "\n")

            # Конец файла
            file.write('''***********************************************************\n''')
            file.write(self.frequency_selection())

    # Выбрана строка таблицы
    def get_clicked_cell(self, row):
        # Запрашиваем из окна, значение порога
        window_width = self.lineEdit_window_width.text()

        # Проверка на цифры и положительность
        if not window_width.isdigit():
            QMessageBox.about(None, "Ошибка ввода", "Введите положительное число в поле ширины окна.")
            return

        window_width = float(window_width)

        frequency_left_or_right = window_width / 2
        # Приближаем область с выделенной частотой
        self.ax1.set_xlim([
            self.data_signals.frequency_peak[row - 1] - frequency_left_or_right,
            self.data_signals.frequency_peak[row - 1] + frequency_left_or_right
        ])

        # Перерисовываем
        self.canvas1.draw()

    # Инициализация пустой таблицы
    def initialize_table(self):
        self.tableWidget_frequency_absorption.setColumnCount(3)
        self.tableWidget_frequency_absorption.setHorizontalHeaderLabels(["Частота МГц", "Гамма", ""])
        self.tableWidget_frequency_absorption.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
        self.tableWidget_frequency_absorption.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)

    # Инициализация пустого верхнего графика
    def initialize_figure(self, fig, ax):
        # Инициализирует фигуру matplotlib внутри контейнера GUI.
        # Вызываем только один раз при инициализации

        # Создание фигуры (self.fig и self.ax)
        self.fig1 = fig
        self.ax1 = ax
        # Создание холста
        self.canvas1 = FigureCanvas(self.fig1)
        self.plotLayout.addWidget(self.canvas1)
        self.canvas1.draw()
        # Создание Toolbar
        self.toolbar1 = NavigationToolbar(self.canvas1, self.plotWindow_empty_and_signal,
                                          coordinates=True)
        self.plotLayout.addWidget(self.toolbar1)

    # Инициализация пустого нижнего графика
    def initialize_figure2(self, fig, ax):
        # Инициализирует фигуру matplotlib внутри контейнера GUI.
        # Вызываем только один раз при инициализации

        # Создание фигуры (self.fig и self.ax)
        self.fig2 = fig
        self.ax2 = ax
        # Создание холста
        self.canvas2 = FigureCanvas(self.fig2)
        self.plotLayout2.addWidget(self.canvas2)
        self.canvas2.draw()
        # Создание Toolbar
        self.toolbar2 = NavigationToolbar(self.canvas2, self.plotWindow_error_and_threshold,
                                          coordinates=True)
        self.plotLayout2.addWidget(self.toolbar2)

    # Чтение и построение сигнала без шума
    def plotting_without_noise(self):
        # Вызов окна выбора файла
        # filename, filetype = QFileDialog.getOpenFileName(None,
        #                                                 "Выбрать файл без шума",
        #                                                 ".",
        #                                                "Spectrometer Data(*.csv);;All Files(*)")
        filename = "25empty.csv"

        # Если имя файла не получено, сброс
        if filename == '':
            return

        # Чтение данных
        with open(filename) as f:
            list_line = f.readlines()  # Читаем по строчно, в список

        if self.checkBox_read_filter.checkState():
            # Считываем "Частоту от"
            start_frequency = self.lineEdit_filter_frequency_start.text()
            # Проверка
            if not frequency_input_check(start_frequency, "Частота от"):
                return
            # Приводим к дробному
            start_frequency = float(start_frequency)

            # Считываем "Частоту до"
            end_frequency = self.lineEdit_filter_frequency_end.text()
            # Проверка
            if not frequency_input_check(end_frequency, "Частота до"):
                return
            # Приводим к дробному
            end_frequency = float(end_frequency)

            # Проверка на правильность границ
            if end_frequency < start_frequency:
                QMessageBox.about(None, "Ошибка ввода", "Частота 'от' больше 'до', в фильтре чтения. ")
                return

            # Парс данных в заданных частотах
            parser(list_line,
                   self.data_signals.empty_frequency, self.data_signals.empty_gamma,
                   start_frequency, end_frequency)
        else:
            # Парс данных
            parser_all_data(list_line,
                            self.data_signals.empty_frequency, self.data_signals.empty_gamma)

        # Очищаем все, что было этого в графике
        self.ax1.clear()
        # Строим данные, добавляем название осей, устанавливаем цвета
        self.ax1.set_xlabel('frequency MHz')
        self.ax1.set_ylabel('gamma')
        self.ax1.plot(self.data_signals.empty_frequency, self.data_signals.empty_gamma, color='r', label='empty')

        # Убеждаемся, что все помещается внутри холста
        self.fig1.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.canvas1.draw()
        self.toolbar1.push_current()

        # Запускаем сценарий: Загружен сигнал без шума
        self.state2_loaded_empty()

    # Чтение и построение полезного сигнала
    def signal_plotting(self):
        # Вызов окна выбора файла
        # filename, filetype = QFileDialog.getOpenFileName(None,
        #                                                 "Выбрать файл сигнала",
        #                                                 ".",
        #                                                 "Spectrometer Data(*.csv);;All Files(*)")

        filename = "25DMSO.csv"

        # Если имя файла не получено, сброс
        if filename == '':
            return

        # Чтение данных
        with open(filename) as f:
            list_line = f.readlines()  # Читаем по строчно, в список

        if self.checkBox_read_filter.checkState():
            # Считываем "Частоту от"
            start_frequency = self.lineEdit_filter_frequency_start.text()
            # Проверка
            if not frequency_input_check(start_frequency, "Частота от"):
                return
            # Приводим к дробному
            start_frequency = float(start_frequency)

            # Считываем "Частоту до"
            end_frequency = self.lineEdit_filter_frequency_end.text()
            # Проверка
            if not frequency_input_check(end_frequency, "Частота до"):
                return
            # Приводим к дробному
            end_frequency = float(end_frequency)

            # Проверка на правильность границ
            if end_frequency < start_frequency:
                QMessageBox.about(None, "Ошибка ввода", "Частота 'от' больше 'до', в фильтре чтения. ")
                return

            # Парс данных в заданных частотах
            parser(list_line,
                   self.data_signals.signal_frequency, self.data_signals.signal_gamma,
                   start_frequency, end_frequency)
        else:
            # Парс данных
            parser_all_data(list_line,
                            self.data_signals.signal_frequency, self.data_signals.signal_gamma)

        # Строим данные
        self.ax1.plot(self.data_signals.signal_frequency, self.data_signals.signal_gamma, color='g', label='signal')
        # Убеждаемся, что все помещается внутри холста
        self.fig1.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.canvas1.draw()
        self.toolbar1.push_current()

        # Запускаем сценарий: Все данные загружены
        self.state3_data_loaded()

    # Разница пустого и полезного сигнала
    def signal_difference(self):
        # Сигналов нет - прекращаем
        if self.data_signals.empty_gamma == [] or self.data_signals.signal_gamma == []:
            return

        width_correlation = self.lineEdit_correlation_width.text()
        # Проверка на цифры и положительность
        if not width_correlation.isdigit():
            QMessageBox.about(None, "Ошибка ввода", "Введите положительное число в поле ширины корреляции.")
            return

        width_correlation = int(width_correlation)

        # Вычитаем отсчеты сигнала с ошибкой и без
        # self.data_signals.difference_empty_and_signal()
        # self.data_signals.correlate_empty_and_signal(width_correlation)
        self.data_signals.correlate_empty_and_signal_pandas2(width_correlation)

        # Отображаем графики
        # очищаем все, что было этого в графике
        self.ax2.clear()
        # Строим данные, добавляем название осей, устанавливаем цвета
        self.ax2.set_xlabel('frequency MHz')
        self.ax2.set_ylabel('error')
        self.ax2.plot(self.data_signals.empty_frequency, self.data_signals.difference, color='g', label='empty')
        # Убеждаемся, что все помещается внутри холста
        self.fig2.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.canvas2.draw()

        # # Запускаем расчет порогового значения, интервалов, отрисовку таблицы
        # self.threshold()

        # Запускаем сценарий: Есть разница сигналов
        self.state4_difference()

    # Расчет порога, интервалов больше порога, частот поглощения, отображение на графиках
    def threshold(self):
        # Нет разницы сигналов - сброс
        if not self.data_signals.difference:
            return

        # Запрос порогового значения
        threshold = self.lineEdit_threshold.text()

        # Проверка ввода
        if not percentage_check(threshold):
            return
        threshold = float(threshold)

        # Значение порога от макс. значения графика ошибки
        threshold_value = threshold

        # ПЕРЕРИСОВКА 2 ГРАФИКА
        self.ax2.clear()
        # Строим данные, добавляем название осей, устанавливаем цвета
        self.ax2.set_xlabel('frequency MHz')
        self.ax2.set_ylabel('error')
        self.ax2.plot(self.data_signals.empty_frequency, self.data_signals.difference, color='g', label='empty')
        # Убеждаемся, что все помещается внутри холста
        self.fig2.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.canvas2.draw()

        # Отрисовка порога
        threshold_signal = [threshold_value] * len(self.data_signals.difference)
        self.ax2.plot(self.data_signals.empty_frequency, threshold_signal, color='r', label='empty')
        self.fig2.tight_layout()
        self.canvas2.draw()

        # ПЕРЕРИСОВКА 1 ГРАФИКА
        self.ax1.clear()
        # Строим данные, добавляем название осей, устанавливаем цвета
        self.ax1.set_xlabel('frequency MHz')
        self.ax1.set_ylabel('gamma')
        self.ax1.plot(self.data_signals.empty_frequency, self.data_signals.empty_gamma, color='r', label='empty')
        self.ax1.plot(self.data_signals.signal_frequency, self.data_signals.signal_gamma, color='g', label='signal')
        # Убеждаемся, что все помещается внутри холста
        self.fig1.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.canvas1.draw()

        # Вычисление промежутков больше порога
        self.data_signals.range_jumps_over_threshold(threshold_value)

        # Выделение промежутков на 1 графике
        for x, y in zip(self.data_signals.frequency_range, self.data_signals.gamma_range):
            self.ax1.plot(x, y, color='b', label='signal')

        # Построение занесенных диапазонов
        self.fig1.tight_layout()
        self.canvas1.draw()

        # Нахождение пиков
        self.data_signals.search_peaks()

        # Вывод данных в таблицу
        self.table()

    # Заполение таблицы
    def table(self):
        # Задаем кол-во столбцов и строк
        self.tableWidget_frequency_absorption.setRowCount(len(self.data_signals.frequency_peak))
        self.tableWidget_frequency_absorption.setColumnCount(3)

        # Задаем название столбцов
        self.tableWidget_frequency_absorption.setHorizontalHeaderLabels(["Частота МГц", "Гамма"])

        # Заполняем таблицу
        index = 0
        for f, g in zip(self.data_signals.frequency_peak, self.data_signals.gamma_peak):
            # значения частоты и гаммы для 0 и 1 столбца
            self.tableWidget_frequency_absorption.setItem(index, 0, QTableWidgetItem(str('%.3f' % f)))
            self.tableWidget_frequency_absorption.setItem(index, 1, QTableWidgetItem(str('%.7E' % g)))

            # Элемент 2 столбца - checkbox, сохранения данных
            check_box = QtWidgets.QCheckBox()  # Создаем объект чекбокс
            check_box.setCheckState(Qt.Checked)  # Задаем состояние - нажат
            check_box.toggled.connect(self.frequency_selection)  # Обработчик нажатия
            self.tableWidget_frequency_absorption.setCellWidget(index, 2, check_box)  # Вводим в таблицу

            index += 1

        # Размеры строк выровнять под содержимое
        self.tableWidget_frequency_absorption.resizeColumnsToContents()
        # Вывести статистику под таблицей
        self.frequency_selection()

    # Сбор и вывод статистики под таблицей
    def frequency_selection(self):
        number_of_selected = 0  # Кол-во выбранных
        number_of_missed = 0  # Кол-во пропущенных
        total = 0  # Кол-во всего

        # Перебираем строки
        for i in range(self.tableWidget_frequency_absorption.rowCount()):
            # Если состояние checkbox - выбрано, то добавляем с выбранным
            if self.tableWidget_frequency_absorption.cellWidget(i, 2).checkState() == Qt.CheckState.Checked:
                number_of_selected += 1
            else:
                number_of_missed += 1
            total += 1

        # Создаем строки статистики
        text_all_number = "Всего: " + str(total)  # Всего элементов
        percentage_of_selected = number_of_selected / total * 100  # Процент выбранных
        text_selected = "Выбрано: " + str(number_of_selected) + " ( " + str('%.2f' % percentage_of_selected) + "% ) "
        text_statistics = text_all_number + ' ' + text_selected  # Итоговая строка

        # Вывод в label под таблицей
        self.label_statistics_on_selected_frequencies.setText(text_statistics)

        # Возвращает текст статистики
        return text_statistics
