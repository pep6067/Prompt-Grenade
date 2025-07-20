# PromptGrenade Factory
![Prompt Grenade Preview](assets/Prompt-Grenade-logo.png)

A lean, legal engine for generating strange prompts that confuse the smartest language models.  
No hacks. No exploits. Just precision-crafted input — and proof that machines still break under pressure.

Built by **Ian Patel**, 15-year-old self-taught AI researcher and founder of multiple AI ventures.  
Running since age 9. Publishing now.

Created for the 'now notorious', **The Feather Light Company**! 

---

## What This Is

PromptGrenade is a modular pipeline that:

- Auto-generates prompts designed to cause logic breakdowns, contradictions, or awkward refusals in LLMs.
- Uses open local models (via Ollama) like `CodeLlama` or `OpenHermes` to manufacture prompt variations.
- Sends these prompts through APIs like Gemini 2.5 Flash, logging how the models respond.
- Tags the failures — not with opinion, but with concrete breakdown types like:
  - `Contradiction`
  - `Incoherent`
  - `Refusal`
  - `Acceptable`

**There are no jailbreaks here.** Every prompt respects API terms and remains within the sandbox.  
The goal: to test how robust models are against linguistic pressure — not to beat the firewall.

---

## What's Included

- `prompt-grenade.py`: Core factory script to generate, test, and tag prompts.
- `/collections`: Collection of Prompt and Answer pairs, categorized and organized.
- `test_1, test_2 & test_3`: Test files containing raw prompts, replies, tags, and other meta-data for the lazy-crew.


### Also Included:

> Three full tests I ran with the system — no need to run the code if you just want to see the impact.  
Check the `test` files for raw prompts, responses, and failure type analysis.

---

## Install & Run

```bash
# Step 1 — Clone the repo
git clone https://github.com/your-username/promptgrenade.git
cd promptgrenade

# Step 2 — Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 3 — Install Ollama + load models
ollama run codellama
# Or: ollama run openhermes

# Step 4 — Run the factory
python promptgrenade.py

# Output will be stored in root folder and in /collections
```
That's it! Thanks for reading it through. Well, I hope you enjoy seeing LLMs break under Prompt Grenade's 'friendly attacks'. 

Psst. If you find something interesting, or just want to collaborate, here's my E-mail: ianpatel5c@gmail.com 
