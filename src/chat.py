"""
chat.py - Interactive Chat REPL

Talk to the Liquid Mamba model in real-time.
Type your prompt, see the model's continuation.
"""

import sys
import torch
import torch.nn.functional as F
from model import LiquidStreamModel, ModelArgs

class ChatBot:
    def __init__(self, max_gen=100, temperature=0.8):
        self.max_gen = max_gen
        self.temperature = temperature

        # Load model
        args = ModelArgs(d_model=64, n_layer=2)
        self.model = LiquidStreamModel(args)
        self.model.eval()

        try:
            self.model.load_state_dict(
                torch.load('model.pt', map_location='cpu', weights_only=True)
            )
            print("✓ Loaded trained model")
        except:
            print("⚠ Using random weights (no model.pt)")

    def generate(self, prompt: str) -> str:
        """Generate continuation from prompt."""
        # Encode prompt as bytes
        tokens = list(prompt.encode('utf-8'))
        context = torch.tensor([tokens], dtype=torch.long)

        generated = []

        with torch.no_grad():
            for _ in range(self.max_gen):
                logits = self.model(context)
                probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                byte_val = next_token.item()
                generated.append(byte_val)

                # Stop on newline for chat-like behavior
                if byte_val == ord('\n') and len(generated) > 5:
                    break

                context = torch.cat([context, next_token], dim=1)
                if context.shape[1] > 256:
                    context = context[:, -256:]

        return bytes(generated).decode('utf-8', errors='replace')

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "=" * 50)
        print("Liquid Analog Stream - Chat Mode")
        print("=" * 50)
        print("Type your prompt and press Enter.")
        print("Commands: /quit, /temp 0.5, /len 200")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith('/'):
                if user_input == '/quit':
                    print("Goodbye!")
                    break
                elif user_input.startswith('/temp '):
                    try:
                        self.temperature = float(user_input.split()[1])
                        print(f"Temperature set to {self.temperature}")
                    except:
                        print("Usage: /temp 0.5")
                elif user_input.startswith('/len '):
                    try:
                        self.max_gen = int(user_input.split()[1])
                        print(f"Max length set to {self.max_gen}")
                    except:
                        print("Usage: /len 200")
                else:
                    print("Unknown command. Try /quit, /temp, /len")
                continue

            # Generate response
            response = self.generate(user_input)
            print(f"Bot: {user_input}{response}")
            print()

def main():
    bot = ChatBot()
    bot.chat()

if __name__ == "__main__":
    main()
