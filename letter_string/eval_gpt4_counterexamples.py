from openai import OpenAI
from letter_string.secrets_openai import OPENAI_KEY




def test_gpt4():
    client = OpenAI(api_key=OPENAI_KEY)
    assistant = client.beta.assistants.create(
        model='gpt-4-0125-preview',
        temperature=0,
        top_p=0,
        tools=[{"type": "code_interpreter"}],  # Added this line
    )

    problem_txts = [
        ('[q a h v] [q a h m]', '[k w b f] [ ? ]', '[k w b t]'),
        ('[n j r q] [n j r h]', '[j r q a] [ ? ]', '[j r q v]'),
        ('[k w b f] [k w b t]', '[y l k w] [ ? ]', '[y l k f]'),
        ('[l k w b] [l k w z]', '[p d i c] [ ? ]', '[p d i e]'),
        ('[f z t n] [f z t r]', '[o p d i] [ ? ]', '[o p d s]'),

        ('[a h v g] [a h v u]', '[f z t n] [ ? ]', '[f z t r]'),
        ('[t n j r] [t n j a]', '[y l k w] [ ? ]', '[y l k f]'),
        ('[t n j r] [t n j a]', '[m u o p] [ ? ]', '[m u o i]'),
        ('[x y l k] [x y l b]', '[u o p d] [ ? ]', '[u o p c]'),
        ('[q a h v] [q a h m]', '[b f z t] [ ? ]', '[b f z j]'),
    ]
    res = []
    for problem_e, problem_t, sol in problem_txts:
        prompt = f"""
		Let’s solve a puzzle problem involving the following fictional alphabet: [x y l k w b f z t n j r q a h v g m u o p d i c s e].

		{problem_e}
		{problem_t}

		Please only provide the answer. Do not provide any additional explanation. 

		Answer:
		"""
        print(prompt)
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            value = messages.data[0].content[0].text.value
            print(value)
            res.append((value, sol))
        else:
            print(run.status)

    print(res)
    print('done')
    #[('kwbz', '[k w b t]'), ('j r q h', '[j r q v]'), ('y l k b', '[y l k f]'), ('p d i s', '[p d i e]'), ('[o p d s]', '[o p d s]'),
    # ('f z t o', '[f z t r]'), ('y l k b', '[y l k f]'), ('m u o d', '[m u o i]'), ('\\[u o p i\\]', '[u o p c]'), ('b f z n', '[b f z j]')]
    # --> 1/10 correct; in paper: 8/10


if __name__ == '__main__':
    test_gpt4()