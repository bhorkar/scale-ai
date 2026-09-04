from matching import match_all, openai_call, openai_embed
from parsing import load_people, load_tasks


def main(people_csv="people.csv", tasks_csv="tasks.csv", llm_call=None, embed_fn=None):
    people = load_people(people_csv)
    tasks = load_tasks(tasks_csv)
    print(f"app start people={len(people)} tasks={len(tasks)}")
    for result in match_all(
        tasks,
        people,
        llm_call or openai_call,
        k=2,
        workers=2,
        batch_size=2,
        embed_fn=embed_fn or openai_embed,
    ):
        print(result)


if __name__ == "__main__":
    main()

