"""Synthetic diagnostic fixtures in ten languages and mixed-language turns.

Hand-authored templates and name variants exercise character offsets, scripts,
known-value matching, and model limitations. These are regression fixtures,
not independently annotated real-world accuracy benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", ".build", "gold")

# --------------------------------------------------------------------------
# Per-language material.  Each entry is (written form, ASR form).
# --------------------------------------------------------------------------

L = {}

L["en"] = {
    "name": "English",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Aisha Hasan", "aisha hassan"),
        ("Tomasz Nowak", "tomash novak"),
        ("Mei-Ling Chen", "may ling chen"),
        ("Darragh O'Neill", "dara oneill"),
    ],
    "parent": [("Farhana Hasan", "farhana hassan"), ("Katarzyna Nowak", "katarina novak")],
    "sibling": [("Zaid", "zade"), ("Niamh", "neev")],
    "teacher": [("Ms Rupa", "miss roopa"), ("Mr Whitfield", "mister whitfield")],
    "doctor": [("Dr Ahmed", "doctor ahmad"), ("Dr Osei", "doctor o say")],
    "school": [
        ("Greenfield Primary School", "green field primary school"),
        ("St Aidan's Academy", "saint aidans academy"),
    ],
    "street": [
        ("21 Oak Street", "twenty one oak street"),
        ("14 Beechwood Avenue", "fourteen beechwood avenue"),
    ],
    "city": [("Leeds", "leeds"), ("Bradford", "bradford")],
    "phone": [("07911 123456", "oh seven nine one one one two three four five six")],
    "email": [("farhana.hasan@gmail.com", "farhana dot hasan at gmail dot com")],
    "dob": [("12/03/2016", "the twelfth of march twenty sixteen")],
    "age": [("nine", "nine"), ("seven", "seven")],
    "lines": [
        ("PARENT", "Right {child}, tell {doctor} how school has been."),
        ("CHILD", "I'm {age} now and I go to {school}."),
        ("CHILD", "My brother {sibling} walks with me from {street}."),
        ("PARENT", "I'm {parent}. My mobile is {phone} or {email}."),
        ("PARENT", "Her date of birth is {dob} and we live in {city}."),
        ("CHILD", "My teacher is {teacher} and she is nice."),
    ],
}

L["es"] = {
    "name": "Spanish",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Lucía Fernández", "lucia fernandes"),
        ("Mateo Ruiz", "mateo ruis"),
        ("Valentina Ortega", "balentina ortega"),
    ],
    "parent": [("Carmen Fernández", "carmen fernandes"), ("Javier Ruiz", "jabier ruis")],
    "sibling": [("Diego", "diego"), ("Inés", "ines")],
    "teacher": [
        ("la señorita Pilar", "la senyorita pilar"),
        ("el señor Álvarez", "el senyor albares"),
    ],
    "doctor": [("la doctora Beltrán", "la doctora beltran"), ("el doctor Sáez", "el doctor saes")],
    "school": [
        ("el colegio Santa Teresa", "el colegio santa teresa"),
        ("la escuela Miguel Hernández", "la escuela miguel ernandes"),
    ],
    "street": [
        ("calle Beniarda 13", "calle beniarda trece"),
        ("avenida Gaspar Aguilar 90", "avenida gaspar aguilar noventa"),
    ],
    "city": [("Valencia", "balencia"), ("Sevilla", "sebilla")],
    "phone": [("612 345 678", "seis uno dos tres cuatro cinco seis siete ocho")],
    "email": [("carmen.fernandez@gmail.com", "carmen punto fernandez arroba gmail punto com")],
    "dob": [("12/03/2016", "el doce de marzo de dos mil dieciséis")],
    "age": [("nueve", "nueve"), ("siete", "siete")],
    "lines": [
        ("PARENT", "Bueno {child}, cuéntale a {doctor} cómo te va en el colegio."),
        ("CHILD", "Tengo {age} años y voy a {school}."),
        ("CHILD", "Mi hermano {sibling} camina conmigo desde {street}."),
        ("PARENT", "Soy {parent}. Mi móvil es {phone} o {email}."),
        ("PARENT", "Su fecha de nacimiento es {dob} y vivimos en {city}."),
        ("CHILD", "Mi maestra es {teacher} y es muy amable."),
    ],
}

L["fr"] = {
    "name": "French",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Léa Marchand", "lea marchant"),
        ("Youssef Benali", "youssef benalli"),
        ("Chloé Lefèvre", "cloe lefevre"),
    ],
    "parent": [("Amélie Marchand", "amelie marchant"), ("Karim Benali", "karim benalli")],
    "sibling": [("Hugo", "ugo"), ("Nadia", "nadya")],
    "teacher": [("madame Dupont", "madame dupond"), ("monsieur Ravel", "monsieur ravelle")],
    "doctor": [
        ("le docteur Sassi", "le docteur sassy"),
        ("la docteure Morel", "la docteure morelle"),
    ],
    "school": [
        ("l'école Jules Ferry", "lecole jules ferry"),
        ("le collège Saint-Exupéry", "le college saint exupery"),
    ],
    "street": [
        ("12 rue des Lilas", "douze rue des lilas"),
        ("5 avenue Victor Hugo", "cinq avenue victor hugo"),
    ],
    "city": [("Lyon", "lion"), ("Roubaix", "roubay")],
    "phone": [("06 12 34 56 78", "zéro six douze trente quatre cinquante six soixante dix huit")],
    "email": [("amelie.marchand@gmail.com", "amelie point marchand arobase gmail point com")],
    "dob": [("12/03/2016", "le douze mars deux mille seize")],
    "age": [("neuf", "neuf"), ("sept", "sept")],
    "lines": [
        ("PARENT", "Alors {child}, raconte à {doctor} comment ça se passe à l'école."),
        ("CHILD", "J'ai {age} ans et je vais à {school}."),
        ("CHILD", "Mon frère {sibling} marche avec moi depuis {street}."),
        ("PARENT", "Je suis {parent}. Mon portable est {phone} ou {email}."),
        ("PARENT", "Sa date de naissance est {dob} et nous habitons à {city}."),
        ("CHILD", "Ma maîtresse est {teacher} et elle est gentille."),
    ],
}

L["de"] = {
    "name": "German",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Lena Schmidt", "lena schmitt"),
        ("Emre Yıldız", "emre yildis"),
        ("Jonas Krüger", "jonas kruger"),
    ],
    "parent": [("Sabine Schmidt", "sabine schmitt"), ("Mehmet Yıldız", "mehmet yildis")],
    "sibling": [("Finn", "fin"), ("Leyla", "leila")],
    "teacher": [("Frau Bergmann", "frau bergman"), ("Herr Vogel", "herr fogel")],
    "doctor": [("Doktor Neumann", "doktor neuman"), ("Frau Doktor Weiß", "frau doktor weiss")],
    "school": [
        ("die Grundschule am Buchenweg", "die grundschule am buchenweg"),
        ("das Gymnasium Sankt Anna", "das gymnasium sankt anna"),
    ],
    "street": [("Buchenweg 14", "buchenweg vierzehn"), ("Lindenstraße 3", "lindenstrasse drei")],
    "city": [("Bochum", "bochum"), ("Freiburg", "freiburg")],
    "phone": [("0151 2345678", "null eins fünf eins zwei drei vier fünf sechs sieben acht")],
    "email": [("sabine.schmidt@gmail.com", "sabine punkt schmidt at gmail punkt com")],
    "dob": [("12.03.2016", "der zwölfte märz zweitausendsechzehn")],
    "age": [("neun", "neun"), ("sieben", "sieben")],
    "lines": [
        ("PARENT", "Also {child}, erzähl {doctor}, wie es in der Schule läuft."),
        ("CHILD", "Ich bin {age} und gehe in {school}."),
        ("CHILD", "Mein Bruder {sibling} läuft mit mir von {street}."),
        ("PARENT", "Ich bin {parent}. Mein Handy ist {phone} oder {email}."),
        ("PARENT", "Ihr Geburtsdatum ist {dob} und wir wohnen in {city}."),
        ("CHILD", "Meine Lehrerin ist {teacher} und sie ist nett."),
    ],
}

L["it"] = {
    "name": "Italian",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Giulia Rossi", "giulia rossy"),
        ("Matteo Esposito", "mateo esposito"),
        ("Sofia Greco", "sofia grego"),
    ],
    "parent": [("Elena Rossi", "elena rossy"), ("Paolo Esposito", "paolo esposito")],
    "sibling": [("Luca", "luca"), ("Chiara", "kiara")],
    "teacher": [
        ("la maestra Bianchi", "la maestra bianki"),
        ("il maestro Conti", "il maestro konti"),
    ],
    "doctor": [
        ("il dottor Marino", "il dottor marrino"),
        ("la dottoressa Fabbri", "la dottoressa fabri"),
    ],
    "school": [
        ("la scuola Alessandro Manzoni", "la scuola alessandro manzoni"),
        ("l'istituto San Giorgio", "listituto san giorgio"),
    ],
    "street": [
        ("via Garibaldi 21", "via garibaldi ventuno"),
        ("corso Vittorio 8", "corso vittorio otto"),
    ],
    "city": [("Bologna", "bolonia"), ("Palermo", "palermo")],
    "phone": [("339 123 4567", "tre tre nove uno due tre quattro cinque sei sette")],
    "email": [("elena.rossi@gmail.com", "elena punto rossi chiocciola gmail punto com")],
    "dob": [("12/03/2016", "il dodici marzo duemilasedici")],
    "age": [("nove", "nove"), ("sette", "sette")],
    "lines": [
        ("PARENT", "Allora {child}, racconta a {doctor} come va a scuola."),
        ("CHILD", "Ho {age} anni e vado a {school}."),
        ("CHILD", "Mio fratello {sibling} cammina con me da {street}."),
        ("PARENT", "Sono {parent}. Il mio cellulare è {phone} oppure {email}."),
        ("PARENT", "La sua data di nascita è {dob} e abitiamo a {city}."),
        ("CHILD", "La mia maestra è {teacher} ed è gentile."),
    ],
}

L["nl"] = {
    "name": "Dutch",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Sanne de Vries", "sanne de vriese"),
        ("Youssra Bakker", "yusra bakker"),
        ("Daan Jansen", "daan janssen"),
    ],
    "parent": [("Marieke de Vries", "marieke de vriese"), ("Peter Jansen", "peter janssen")],
    "sibling": [("Bram", "bram"), ("Fenna", "fenna")],
    "teacher": [("juf Willemsen", "juf willemse"), ("meester Groot", "meester groot")],
    "doctor": [("dokter Visser", "dokter fisser"), ("dokter El Amrani", "dokter el amrany")],
    "school": [
        ("basisschool De Regenboog", "basisschool de regenboog"),
        ("het Sint-Janscollege", "het sint janscollege"),
    ],
    "street": [("Lindenlaan 14", "lindenlaan veertien"), ("Kerkstraat 3", "kerkstraat drie")],
    "city": [("Utrecht", "utrech"), ("Tilburg", "tilburg")],
    "phone": [("06 12345678", "nul zes een twee drie vier vijf zes zeven acht")],
    "email": [("marieke.devries@gmail.com", "marieke punt devries apenstaartje gmail punt com")],
    "dob": [("12-03-2016", "twaalf maart tweeduizend zestien")],
    "age": [("negen", "negen"), ("zeven", "zeven")],
    "lines": [
        ("PARENT", "Goed {child}, vertel {doctor} hoe het op school gaat."),
        ("CHILD", "Ik ben {age} en ik zit op {school}."),
        ("CHILD", "Mijn broer {sibling} loopt met mij mee vanaf {street}."),
        ("PARENT", "Ik ben {parent}. Mijn mobiel is {phone} of {email}."),
        ("PARENT", "Haar geboortedatum is {dob} en we wonen in {city}."),
        ("CHILD", "Mijn juf is {teacher} en ze is aardig."),
    ],
}

L["pt"] = {
    "name": "Portuguese",
    "script": "Latin",
    "cased": True,
    "child": [
        ("Beatriz Almeida", "beatris almeida"),
        ("Miguel Fonseca", "miguel fonseka"),
        ("Inês Carvalho", "ines carvaio"),
    ],
    "parent": [("Cláudia Almeida", "claudia almeida"), ("Rui Fonseca", "rui fonseka")],
    "sibling": [("Tomás", "tomas"), ("Matilde", "matilde")],
    "teacher": [
        ("a professora Sousa", "a professora souza"),
        ("o professor Pinto", "o professor pinto"),
    ],
    "doctor": [("o doutor Faria", "o doutor faria"), ("a doutora Nunes", "a doutora nunes")],
    "school": [
        ("a escola Dom Pedro", "a escola dom pedro"),
        ("o colégio São Vicente", "o colegio sao vicente"),
    ],
    "street": [
        ("rua das Flores 21", "rua das flores vinte e um"),
        ("avenida da Liberdade 5", "avenida da liberdade cinco"),
    ],
    "city": [("Braga", "braga"), ("Setúbal", "setubal")],
    "phone": [("912 345 678", "nove um dois três quatro cinco seis sete oito")],
    "email": [("claudia.almeida@gmail.com", "claudia ponto almeida arroba gmail ponto com")],
    "dob": [("12/03/2016", "doze de março de dois mil e dezasseis")],
    "age": [("nove", "nove"), ("sete", "sete")],
    "lines": [
        ("PARENT", "Então {child}, conta a {doctor} como tem corrido a escola."),
        ("CHILD", "Tenho {age} anos e ando na {school}."),
        ("CHILD", "O meu irmão {sibling} vem comigo desde {street}."),
        ("PARENT", "Sou {parent}. O meu telemóvel é {phone} ou {email}."),
        ("PARENT", "A data de nascimento dela é {dob} e vivemos em {city}."),
        ("CHILD", "A minha professora é {teacher} e é simpática."),
    ],
}

L["ru"] = {
    "name": "Russian",
    "script": "Cyrillic",
    "cased": True,
    "child": [
        ("Алина Соколова", "алина сокалова"),
        ("Тимур Гафуров", "тимур гафуроф"),
        ("Даша Петрова", "даша питрова"),
    ],
    "parent": [("Ольга Соколова", "ольга сокалова"), ("Рустам Гафуров", "рустам гафуроф")],
    "sibling": [("Миша", "миша"), ("Катя", "катя")],
    "teacher": [("Мария Ивановна", "мария ивановна"), ("Сергей Петрович", "сергей петрович")],
    "doctor": [("доктор Лебедев", "доктор лебедеф"), ("доктор Кузнецова", "доктор кузницова")],
    "school": [
        ("школа номер сорок два", "школа номер сорок два"),
        ("гимназия имени Пушкина", "гимназия имени пушкина"),
    ],
    "street": [
        ("улица Лесная 14", "улица лесная четырнадцать"),
        ("проспект Мира 3", "проспект мира три"),
    ],
    "city": [("Казань", "казань"), ("Самара", "самара")],
    "phone": [("8 916 123 45 67", "восемь девять один шесть один два три четыре пять шесть семь")],
    "email": [("olga.sokolova@gmail.com", "ольга точка соколова собака гмейл точка ком")],
    "dob": [("12.03.2016", "двенадцатое марта две тысячи шестнадцатого года")],
    "age": [("девять", "девять"), ("семь", "семь")],
    "lines": [
        ("PARENT", "Так, {child}, расскажи {doctor}, как дела в школе."),
        ("CHILD", "Мне {age} лет, я хожу в {school}."),
        ("CHILD", "Мой брат {sibling} ходит со мной от дома, {street}."),
        ("PARENT", "Я {parent}. Мой телефон {phone} или {email}."),
        ("PARENT", "Её дата рождения {dob}, мы живём в городе {city}."),
        ("CHILD", "Моя учительница {teacher}, она добрая."),
    ],
}

L["hi"] = {
    "name": "Hindi",
    "script": "Devanagari",
    "cased": False,
    "child": [("आयशा हसन", "आइशा हसन"), ("रोहन पिल्लई", "रोहन पिलई"), ("सना क़ुरैशी", "सना कुरेशी")],
    "parent": [("फ़रहाना हसन", "फरहाना हसन"), ("मीरा पिल्लई", "मीरा पिलई")],
    "sibling": [("ज़ैद", "जैद"), ("अंजलि", "अंजली")],
    "teacher": [("रूपा मैडम", "रुपा मैडम"), ("शर्मा सर", "शरमा सर")],
    "doctor": [("डॉक्टर अहमद", "डाक्टर अहमद"), ("डॉक्टर वर्मा", "डाक्टर वरमा")],
    "school": [
        ("ग्रीनफ़ील्ड प्राइमरी स्कूल", "ग्रीनफील्ड प्राइमरी स्कूल"),
        ("सरस्वती विद्या मंदिर", "सरसवती विद्या मंदिर"),
    ],
    "street": [("ओक स्ट्रीट 21", "ओक स्ट्रीट इक्कीस"), ("गांधी नगर 14", "गांधी नगर चौदह")],
    "city": [("लखनऊ", "लखनऊ"), ("इंदौर", "इंदौर")],
    "phone": [("98765 43210", "नौ आठ सात छह पांच चार तीन दो एक शून्य")],
    "email": [("farhana.hasan@gmail.com", "फरहाना डॉट हसन ऐट जीमेल डॉट कॉम")],
    "dob": [("12/03/2016", "बारह मार्च दो हज़ार सोलह")],
    "age": [("नौ", "नौ"), ("सात", "सात")],
    "lines": [
        ("PARENT", "हाँ {child}, {doctor} को बताओ स्कूल कैसा चल रहा है।"),
        ("CHILD", "मैं {age} साल का हूँ और {school} में पढ़ता हूँ।"),
        ("CHILD", "मेरा भाई {sibling} मेरे साथ {street} से चलता है।"),
        ("PARENT", "मैं {parent} हूँ। मेरा नंबर {phone} है या {email}।"),
        ("PARENT", "उसकी जन्म तिथि {dob} है और हम {city} में रहते हैं।"),
        ("CHILD", "मेरी टीचर {teacher} हैं और वो अच्छी हैं।"),
    ],
}

L["ja"] = {
    "name": "Japanese",
    "script": "Japanese",
    "cased": False,
    "spaced": False,
    "child": [("田中さくら", "田中桜"), ("鈴木ひなた", "鈴木日向"), ("佐藤はると", "佐藤陽翔")],
    "parent": [("田中美咲", "田中みさき"), ("鈴木健一", "鈴木けんいち")],
    "sibling": [("ゆうと", "悠斗"), ("あおい", "葵")],
    "teacher": [("山本先生", "山元先生"), ("小林先生", "小林せんせい")],
    "doctor": [("加藤医師", "加藤いし"), ("中村先生", "中村せんせい")],
    "school": [("みどり小学校", "緑小学校"), ("さくら幼稚園", "桜幼稚園")],
    "street": [
        ("桜町3丁目14番", "さくらちょう三丁目十四番"),
        ("本町1丁目21番", "ほんちょう一丁目二十一番"),
    ],
    "city": [("横浜市", "よこはま市"), ("福岡市", "ふくおか市")],
    "phone": [("090-1234-5678", "ゼロ九〇の一二三四の五六七八")],
    "email": [("misaki.tanaka@gmail.com", "みさき ドット たなか アット じーめーる ドット こむ")],
    "dob": [("2016年3月12日", "二千十六年三月十二日")],
    "age": [("九歳", "九さい"), ("七歳", "七さい")],
    "lines": [
        ("PARENT", "{child}、{doctor}に学校のことを話してね。"),
        ("CHILD", "わたしは{age}で、{school}に通っています。"),
        ("CHILD", "兄の{sibling}と{street}から一緒に歩きます。"),
        ("PARENT", "{parent}です。携帯は{phone}、メールは{email}です。"),
        ("PARENT", "娘の生年月日は{dob}で、{city}に住んでいます。"),
        ("CHILD", "担任は{teacher}で、やさしいです。"),
    ],
}

SLOT_ENTITY = {
    "child": "CHILD_NAME",
    "parent": "PARENT_NAME",
    "sibling": "SIBLING_NAME",
    "teacher": "TEACHER_NAME",
    "doctor": "DOCTOR_NAME",
    "school": "SCHOOL_NAME",
    "street": "STREET_ADDRESS",
    "city": "CITY",
    "phone": "PHONE",
    "email": "EMAIL",
    "dob": "DATE_OF_BIRTH",
    "age": "EXACT_AGE",
}

_PUNCT = ".,?!;:。、！？"


def asr_plain(text, lang):
    """Drop punctuation and case the way a recogniser does, without disturbing
    the spaces that separate a slot from the words around it."""
    out = text
    for ch in _PUNCT:
        out = out.replace(ch, "" if lang == "ja" else " ")
    if L[lang].get("cased", True):
        out = out.lower()
    # punctuation became a space, so a comma before a name still separates it
    lead = " " if out[:1].isspace() else ""
    trail = " " if out[-1:].isspace() else ""
    if lang == "ja":
        return "".join(out.split())
    body = " ".join(out.split())
    if not body:
        return " " if (lead or trail) else ""
    return lead + body + trail


def build_document(lang, rnd, asr):
    spec = L[lang]
    cast = {slot: rnd.choice(spec[slot]) for slot in SLOT_ENTITY}
    turns, spans = [], []
    cursor = 0
    for speaker, template in spec["lines"]:
        text, local_spans = "", []
        rest = template
        while "{" in rest:
            pre, _, after = rest.partition("{")
            slot, _, rest = after.partition("}")
            surface = cast[slot][1 if asr else 0]
            pre = asr_plain(pre, lang) if asr else pre
            text += pre
            local_spans.append((len(text), len(text) + len(surface), SLOT_ENTITY[slot]))
            text += surface
        tail = asr_plain(rest, lang) if asr else rest
        text += tail
        text = text.strip()
        for start, end, entity in local_spans:
            spans.append({"start": cursor + start, "end": cursor + end, "entity": entity})
        turns.append({"speaker": speaker, "text": text})
        cursor += len(text) + 1
    known = {
        "CHILD_NAME": [cast["child"][0]],
        "SCHOOL_NAME": [cast["school"][0]],
        "PARENT_NAME": [cast["parent"][0]],
    }
    return {"turns": turns, "spans": spans, "known_values": known, "language": lang}


def build_codeswitched(rnd, asr, languages=("es", "fr", "de", "hi", "ru")):
    """An English conversation with the family's own language mixed in - the case
    Deepgram's `multi` model exists for, and the case a UK family service meets
    every day."""
    other = rnd.choice([language for language in languages if language in L])
    en_doc = build_document("en", rnd, asr)
    other_doc = build_document(other, rnd, asr)
    turns, spans, cursor = [], [], 0
    pairs = []
    for i in range(max(len(en_doc["turns"]), len(other_doc["turns"]))):
        src = en_doc if i % 2 == 0 else other_doc
        if i < len(src["turns"]):
            pairs.append((src, i))
    # rebuild offsets over the interleaved turns
    for src, i in pairs:
        turn = src["turns"][i]
        start_of_turn = sum(len(src["turns"][k]["text"]) + 1 for k in range(i))
        end_of_turn = start_of_turn + len(turn["text"])
        for sp in src["spans"]:
            if start_of_turn <= sp["start"] < end_of_turn:
                spans.append(
                    {
                        "start": cursor + (sp["start"] - start_of_turn),
                        "end": cursor + (sp["end"] - start_of_turn),
                        "entity": sp["entity"],
                    }
                )
        turns.append(dict(turn))
        cursor += len(turn["text"]) + 1
    known = dict(en_doc["known_values"])
    for k, v in other_doc["known_values"].items():
        known.setdefault(k, []).extend(v)
    return {"turns": turns, "spans": spans, "known_values": known, "language": "en+" + other}


def write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(per_language=60, seed=23):
    os.makedirs(OUT, exist_ok=True)
    index = {}
    for lang in L:
        for condition in ("clean", "asr"):
            rnd = random.Random(
                seed + int.from_bytes(hashlib.sha256(lang.encode()).digest()[:4], "big")
            )
            recs = []
            for i in range(per_language):
                doc = build_document(lang, rnd, asr=(condition == "asr"))
                doc["doc_id"] = f"ml-{lang}-{condition}-{i:03d}"
                recs.append(doc)
            path = os.path.join(OUT, f"ml_{lang}_{condition}.jsonl")
            write(path, recs)
            index[f"ml_{lang}_{condition}"] = len(recs)
    for condition in ("clean", "asr"):
        rnd = random.Random(seed + 7)
        recs = []
        for i in range(per_language):
            doc = build_codeswitched(rnd, asr=(condition == "asr"))
            doc["doc_id"] = f"ml-cs-{condition}-{i:03d}"
            recs.append(doc)
        write(os.path.join(OUT, f"ml_codeswitch_{condition}.jsonl"), recs)
        index[f"ml_codeswitch_{condition}"] = len(recs)
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
