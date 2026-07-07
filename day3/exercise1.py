class PromptManager:
    def summarize(self, text):
        return f"Summary: {text[:50]}..."
    def translate(self, text):
        return f"Translation: {text[:50]}..."
    def Explain(self, text):
        return f"Explanation: {text[:50]}..."

pm = PromptManager()
choice = input("Enter your choice: ")
if choice == "1":       # summarize
    text = input("Enter some text: ")
    result = pm.summarize(text)
    print(result)
elif choice == "2":     # translate
    text = input("Enter some text: ")
    result = pm.translate(text)
    print(result)
elif choice == "3":     # explain                   
    text = input("Enter some text: ")
    result = pm.Explain(text)
    print(result)
else:
    print("Invalid choice")