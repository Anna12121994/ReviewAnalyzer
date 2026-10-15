
import csv
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline



PROJECT_DIR = Path(__file__).resolve().parent

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


minilm = None
roberta = None
reference_vectors = None



examples = {
    "Позитивний": [
        "The product is excellent and works perfectly.",
        "I love this product and I am very satisfied.",
        "The service was amazing and the quality is great.",
        "The application is fast, reliable and easy to use."
    ],
    "Негативний": [
        "The product is terrible and does not work.",
        "I am disappointed with the quality.",
        "The service was awful and I am unhappy.",
        "The application crashes and works very slowly."
    ],
    "Нейтральний": [
        "The product is average and nothing special.",
        "The service was okay.",
        "The product works as expected.",
        "The product has both advantages and disadvantages."
    ]
}



reviews = [
    "The application works perfectly. I use it every day.",
    "The delivery was late, and the package was damaged.",
    "The product is acceptable, but nothing special.",
    "Excellent customer support and fast service!",
    "The software crashes every time I open it.",
    "The headphones have clear sound and comfortable ear cushions.",
    "I am disappointed with the quality of this product.",
    "The website is easy to navigate and looks modern.",
    "The food arrived cold and tasted terrible.",
    "The staff were friendly and helped me solve the problem.",
    "The battery lasts much longer than I expected.",
    "This is the worst purchase I have made this year.",
    "The service was fine, but the price was a little high.",
    "Setup was simple, and everything worked on the first try.",
    "The instructions are confusing and difficult to follow.",
    "I love the design, and the product feels very durable.",
    "The order arrived two days earlier than expected.",
    "The camera quality is poor in low light.",
    "Nothing was wrong with the item, but it did not impress me.",
    "Customer support ignored my messages for several days.",
    "The new update fixed the bugs and improved performance.",
    "The chair is uncomfortable after sitting for an hour.",
    "The subscription is affordable and offers great value.",
    "The app is slow and drains my phone battery.",
    "The hotel room was clean, quiet, and comfortable.",
    "The product stopped working after only one week.",
    "The packaging was neat, and the item matched the description.",
    "The meal was okay, although the portions were small.",
    "I received the wrong item and had to return it.",
    "The course was informative, engaging, and well organized.",
    "The printer works as expected, with no major issues.",
    "I would not recommend this service to anyone.",
    "The screen is bright, sharp, and easy to read.",
    "The repair took longer than promised, but the result is good.",
    "The price is reasonable for the features provided.",
    "The instructions were clear, and installation took ten minutes.",
    "The sound quality is average, and the controls feel cheap.",
    "I had a wonderful experience from start to finish.",
    "The product looks nice, but it does not work properly.",
    "The support team resolved my issue quickly and professionally."
]



special_reviews = [
    ("Короткий позитивний", "Great!"),
    ("Короткий негативний", "Terrible!"),
    ("Короткий нейтральний", "It works."),
    (
        "Довгий позитивний",
        "I purchased this application last month. "
        "The interface is simple, the performance is excellent, "
        "and customer support is very helpful. "
        "I use the application every day and recommend it."
    ),
    (
        "Довгий негативний",
        "I ordered this product two weeks ago. "
        "The delivery was late, the packaging was damaged, "
        "and the product stopped working after one day. "
        "Customer service did not respond to my messages."
    ),
    (
        "Саркастичний",
        "Wonderful! The app crashed five times today."
    ),
    (
        "Саркастичний",
        "Great, another update that breaks everything. Amazing!"
    ),
    (
        "Суперечливий",
        "The design is beautiful, but the battery is terrible."
    ),
    (
        "Суперечливий",
        "The delivery was slow, but the product works perfectly."
    )
]



def load_model(model_number):
    global minilm, roberta, reference_vectors

    if model_number == "1":
        if minilm is None:
            print("\nЗавантаження MiniLM...")
            minilm = SentenceTransformer(MINILM_NAME)

            reference_texts = [
                text
                for texts in examples.values()
                for text in texts
            ]

            reference_vectors = minilm.encode(
                reference_texts,
                convert_to_tensor=True
            )

            print("MiniLM готова!")

    elif model_number == "2":
        if roberta is None:
            print("\nЗавантаження RoBERTa...")

            roberta = pipeline(
                "sentiment-analysis",
                model=ROBERTA_NAME,
                framework="pt",
                device=-1
            )

            print("RoBERTa готова!")

    else:
        raise ValueError("Невідома модель")



def validate_text(text):
    if not text.strip():
        raise ValueError("Відгук не може бути порожнім!")

    if not any(char.isalpha() for char in text):
        raise ValueError("Введіть текст, а не тільки цифри!")



def analyze_minilm(texts):
    load_model("1")

    vectors = minilm.encode(
        texts,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        vectors,
        reference_vectors
    )

    results = []

    for text, row in zip(texts, similarities):
        scores = {}
        index = 0

        for label, reference_texts in examples.items():
            count = len(reference_texts)

            scores[label] = max(
                row[index:index + count]
            ).item()

            index += count

        sentiment = max(scores, key=scores.get)

        results.append({
            "text": text,
            "sentiment": sentiment,
            "score": round(scores[sentiment] * 100, 2)
        })

    return results


# Аналіз списку за допомогою RoBERTa
def analyze_roberta(texts):
    load_model("2")

    predictions = roberta(
        texts,
        truncation=True,
        max_length=512,
        batch_size=8
    )

    labels = {
        "positive": "Позитивний",
        "negative": "Негативний",
        "neutral": "Нейтральний"
    }

    results = []

    for text, prediction in zip(texts, predictions):
        sentiment = labels.get(
            prediction["label"].lower(),
            prediction["label"]
        )

        results.append({
            "text": text,
            "sentiment": sentiment,
            "score": round(prediction["score"] * 100, 2)
        })

    return results


# Загальна функція аналізу
def analyze(texts, model_number):
    for text in texts:
        validate_text(text)

    if model_number == "1":
        return analyze_minilm(texts)

    if model_number == "2":
        return analyze_roberta(texts)

    raise ValueError("Неправильний номер моделі!")


# Виведення результатів
def print_results(results, model_number):
    score_name = (
        "Схожість" if model_number == "1"
        else "Впевненість моделі"
    )

    print("\nРЕЗУЛЬТАТИ")

    for number, result in enumerate(results, 1):
        print(f"\nВідгук №{number}")
        print(f"Текст: {result['text']}")
        print(f"Тональність: {result['sentiment']}")
        print(f"{score_name}: {result['score']:.2f}%")
        print("-" * 50)


# Статистика
def show_statistics(results):
    counts = Counter(
        result["sentiment"] for result in results
    )

    print("\ СТАТИСТИКА")
    print(f"Усього відгуків: {len(results)}")
    print(f"Позитивних: {counts['Позитивний']}")
    print(f"Негативних: {counts['Негативний']}")
    print(f"Нейтральних: {counts['Нейтральний']}")




def save_csv(filename, headers, rows):
    path = PROJECT_DIR / filename

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\nФайл збережено: {path}")



def save_analysis(results, model_number, filename):
    score_name = (
        "Similarity (%)" if model_number == "1"
        else "Confidence (%)"
    )

    rows = [
        [
            number,
            result["text"],
            result["sentiment"],
            result["score"]
        ]
        for number, result in enumerate(results, 1)
    ]

    save_csv(
        filename,
        ["Number", "Review", "Sentiment", score_name],
        rows
    )


# Вибір моделі
def choose_model():
    print("\nОберіть модель:")
    print("1 - MiniLM (модель викладача)")
    print("2 - RoBERTa (класифікатор тональності)")

    choice = input("Ваш вибір: ").strip()

    if choice not in ("1", "2"):
        print("Помилка: оберіть 1 або 2!")
        return None

    return choice


def single_review():
    model_number = choose_model()

    if model_number is None:
        return

    text = input("\nВведіть відгук англійською: ").strip()

    validate_text(text)

    results = analyze([text], model_number)

    print_results(results, model_number)

    save_analysis(
        results,
        model_number,
        "single_result.csv"
    )


# Аналіз усіх 40 відгуків
def analyze_all_reviews():
    model_number = choose_model()

    if model_number is None:
        return

    results = analyze(reviews, model_number)

    print_results(results, model_number)
    show_statistics(results)

    save_analysis(
        results,
        model_number,
        f"results_model_{model_number}.csv"
    )



def compare_models():
    print("\n========== ПОРІВНЯННЯ МОДЕЛЕЙ ==========")

    results1 = analyze(reviews, "1")
    results2 = analyze(reviews, "2")

    rows = []
    differences = 0

    for number, (r1, r2) in enumerate(
        zip(results1, results2), 1
    ):
        different = r1["sentiment"] != r2["sentiment"]

        if different:
            differences += 1

        print(f"\nВідгук №{number}")
        print(f"Текст: {r1['text']}")
        print(
            f"MiniLM: {r1['sentiment']} "
            f"(схожість {r1['score']:.2f}%)"
        )
        print(
            f"RoBERTa: {r2['sentiment']} "
            f"(впевненість {r2['score']:.2f}%)"
        )

        if different:
            print("Моделі дали різні відповіді!")

        rows.append([
            number,
            r1["text"],
            r1["sentiment"],
            r1["score"],
            r2["sentiment"],
            r2["score"],
            "Так" if different else "Ні"
        ])

    print("\n ПІДСУМОК ")
    print(f"Усього відгуків: {len(reviews)}")
    print(f"Різних відповідей: {differences}")

    save_csv(
        "comparison.csv",
        [
            "Number",
            "Review",
            "MiniLM Sentiment",
            "MiniLM Similarity (%)",
            "RoBERTa Sentiment",
            "RoBERTa Confidence (%)",
            "Different"
        ],
        rows
    )

    print("\nСтатистика MiniLM:")
    show_statistics(results1)

    print("\nСтатистика RoBERTa:")
    show_statistics(results2)



def special_tests():
    print("\n СКЛАДНІ ВІДГУКИ ")

    texts = [text for category, text in special_reviews]

    results1 = analyze(texts, "1")
    results2 = analyze(texts, "2")

    rows = []

    for number, (item, r1, r2) in enumerate(
        zip(special_reviews, results1, results2), 1
    ):
        category, text = item

        print(f"\nТип: {category}")
        print(f"Текст: {text}")
        print(
            f"MiniLM: {r1['sentiment']} "
            f"({r1['score']:.2f}% схожості)"
        )
        print(
            f"RoBERTa: {r2['sentiment']} "
            f"({r2['score']:.2f}% впевненості)"
        )

        rows.append([
            number,
            category,
            text,
            r1["sentiment"],
            r1["score"],
            r2["sentiment"],
            r2["score"]
        ])

    save_csv(
        "special_tests.csv",
        [
            "Number",
            "Type",
            "Review",
            "MiniLM Sentiment",
            "MiniLM Similarity (%)",
            "RoBERTa Sentiment",
            "RoBERTa Confidence (%)"
        ],
        rows
    )


# Головне меню
def main():
    while True:

        print("           REVIEW ANALYZER")
        print("1 - Аналіз одного відгуку")
        print("2 - Аналіз усіх 40 відгуків")
        print("3 - Порівняння двох моделей")
        print("4 - Тестування складних відгуків")
        print("0 - Вихід")

        choice = input("\nОберіть пункт: ").strip()

        try:
            if choice == "1":
                single_review()

            elif choice == "2":
                analyze_all_reviews()

            elif choice == "3":
                compare_models()

            elif choice == "4":
                special_tests()

            elif choice == "0":
                print("\nРоботу програми завершено!")
                break

            else:
                print("Невірний пункт меню!")

        except (ValueError, OSError, RuntimeError) as error:
            print(f"\nПомилка: {error}")
            print("Перевірте введення, бібліотеки та інтернет.")

        except Exception as error:
            print(f"\nНеочікувана помилка: {error}")


if __name__ == "__main__":
    main()
