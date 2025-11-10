# /main.py

from core.agents import InvestmentAgent


def main():
    print("Welcome to your AI Investment Assistant!")
    print("Type 'exit' to end the conversation.")

    investment_agent = InvestmentAgent()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        response = investment_agent.run(user_input)
        print(f"AI Assistant: {response}")


if __name__ == "__main__":
    main()
