"""Связи между записями: общие коды/термины, порядок обработки."""
from laim_rca.related import find_related, processing_order, terms


def rec(trace, query, answer=''):
    return {'trace_id': trace, 'user_query': query, 'agent_response': answer}


def test_terms_keep_codes_and_drop_filler():
    assert terms('Напиши по каким кодам комиссии ОФР 47109.99') == {'комиссии', 'офр', '47109.99'}
    assert 'dfa_oper_fee_sc_wd' in terms('по коду комиссии DFA_OPER_FEE_SC_WD есть НДС?')
    assert terms('Что такое такое?') == set()


def test_links_need_a_code_or_two_words():
    # регистр не важен: «крюл» в вопросе — «КРЮЛ» в чужом ответе
    assert find_related([rec('t1', 'Что такое крюл?'), rec('t2', 'Статус', 'Ошибки у фабрики КРЮЛ')])[0] == [1]
    # два общих значимых слова — связь
    assert find_related([rec('t3', 'Как исправить ошибку NOT_FOUND?'),
                         rec('t4', 'Код 47109', 'ошибку сети не исправить')])[0] == [1]
    # одно общее слово — не связь
    assert find_related([rec('t3', 'Как исправить ошибку NOT_FOUND?'),
                         rec('t5', 'Код 47110', 'Сбой: не исправить')])[0] == []


def test_words_common_to_most_records_do_not_link():
    records = [rec(f't{i}', f'Отчёт ведомость {i}', 'готово') for i in range(6)]
    assert find_related(records) == [[]] * 6       # тема всего агента, а не связь


def test_same_trace_records_are_always_linked_first():
    records = [rec('t1', 'вопрос один'), rec('t9', 'ГБК ГБК'), rec('t1', 'совсем другое'), rec('t8', 'ГБК')]
    related = find_related(records)
    assert related[0][0] == 2 and related[1] == [3]


def test_processing_order_keeps_related_together():
    records = [rec('a', 'ГБК?'), rec('b', 'счёт'), rec('c', 'ГБК это что'), rec('d', 'КАФО'), rec('e', 'КАФО нужна')]
    order = processing_order(records, find_related(records))
    assert order == [0, 2, 1, 3, 4] and sorted(order) == list(range(5))
