# Спам-фильтр

### Цель
Создать программу на языке Python, которая будет определять, является ли текст (электронное письмо) спамом

### План
1. Собрать спам-письма
2. Найти важные слова в спам-письмах с помощью алгоритма tf-idf
3. Сделать то же самое для обычных писем
4. С помощью наивного байесовского классификатора научиться определять, является ли данный текст спамом (сравнивая вероятности, что это спам и что это не спам)
5. Учесть дополнительные параметры (пунктуация, наличие приветствия и обращения по имени)
6. Визуализировать (сделать веб-приложение)

### Что мы сделали
В целом то, что планировали

### Проблемы
:negative_squared_cross_mark: Не нашли много хорошего спама, поэтому результат работы алгоритма может быть неточным
:negative_squared_cross_mark: Непонятно, сколько должны весить дополнительные параметры
:negative_squared_cross_mark: Наличие приветствия определяется закрытым списком слов
:negative_squared_cross_mark: При подсчете слов в собранных письмах каждый раз выдаются разные результаты 
:negative_squared_cross_mark: Веб-приложение существует только на нашем компьютере

### Разработчики
Берлин Влада, Маринина Валерия
