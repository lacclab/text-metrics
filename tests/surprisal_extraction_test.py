from text_metrics.utils import get_surprisal, init_tok_n_model, clean_text


def main():
    # import the content of tests/below_2048.txt as string
    with open("tests/below_2048.txt", "r") as file:
        below_2048 = file.read()

    with open("tests/over_2048.txt", "r") as file:
        over_2048 = file.read()

    model_names = ["facebook/opt-350m", "EleutherAI/pythia-70m"]

    for model_name in model_names:
        tokenizer, model = init_tok_n_model(model_name)
        print(f"Model: {model_name}")
        _ = get_surprisal(clean_text(below_2048), tokenizer, model, model_name, context_stride=1625)
        _ = get_surprisal(clean_text(over_2048), tokenizer, model, model_name, context_stride=1625)
        print("Didn't crash")


if __name__ == "__main__":
    main()
