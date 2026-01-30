### **BETA TEST PLAN – TEMPLATE**

## **1. Project context**
HuRI is a research-driven platform that has the objective to act as a universal middleware for speech and movement for any various physical or digital human embodiments. The focus on our part is to have embodiments as human as possible, regarding the emotion aspect, and try to go beyond the uncanny valley. The goal is to create a modular ecosystem where any combination of AI can control any robot or any virtual avatar. The project focuses on a highly stable architecture, scalable cloud deployment, and innovative emotional expression through motion and voice.

## **2. User role**

| **Role Name** | **Description** |
|---------------|-----------------|
| Library Maintainer | Manages the core Python package, handles versioning, and ensures the stability of the base classes and API. |
| Module Developer | Uses the library to build and package custom plugins or specialized modules for specific use cases. |
| DevOps Engineer | Handles the containerization, orchestration, and distribution of the library across different network nodes or edge devices. |
| Client | User of the library |


## **3. Feature table**

| **Feature ID** | **User role** | **Feature name** | **Short description** |
|--------------|---------------|-------------------------|--------------------------------------|
| F1 | Everyone | Create Module | The lib allows to create a module and use it. |
| F2 | Everyone | Config file | Config file to launch different module combinations.  |
| F3 | Everyone | Parallelism | HuRI can run on 1 to N machines, to split computation and balance payload. |
| F4 | Everyone | Multi-client | HuRI can be used on 1 to N robots with separated discussions. |
| F5 | Everyone | Module - MIC | User can talk to HuRI through microphone. |
| F6 | Everyone | Module - SPK | HuRI can talk to User through speakers. |
| F7 | Everyone | Module - STT | HuRI can transcribe speech into text. |
| F8 | Everyone | Module - INP | User can chat to HuRI through terminal. |
| F9 | Everyone | Module - OUT | HuRI can chat to User through terminal. |
| F10 | Everyone | Module - MOD | HuRI has 3 modes: Discussion, Inserting context & Inserting information. User can switch between modes. |
| F11 | Everyone | Module - TTS | HuRI can generate speech with text. Generate audio. |
| F12 | Everyone | Module - MOV | HuRI can generate body movement by putting points in space |
| F13 | Everyone | Module - RAG | HuRI can retrieve text from files, saved texts, old conversations, etc. |
| F14 | Everyone | Module - LLM | HuRI can generate text from a given context |
| F15 | Everyone | Module - TAN | User speech text will be analysed to understand their emotion. |
| F16 | Everyone | Module - EIN | Analysed emotion is mixed with the context. |
| F17 | Everyone | Module - AMM  | HuRI will store and manage an artificial memory |

---

## **4. Success Criteria**

| **Feature ID** | **Key success criteria** | **Indicator/metric** | **Result** |
|--------------|---------------------------------------|-----------------------|----------------|
| F1 | Creating various Modules. | 20 attempts -- expected 100% | Achieved (/20) |
| F2 | Launching different scenarios with different config files. | 5 files -- expected 100% | Achieved (/5) |
| F3 - F4 | Running 1 or several modules on 1 or several machines. | All modules running on 1 machine. All modules running on different machines. Several modules running on several machines. | Scenario achieved (/3) |
| F5 | Using a microphone and being recorded. | 10 messages recorded over 3 different devices -- expected 100% | Files recorded (/30) |
| F6 | Emitting sound through speakers. | 10 audio files played on 3 different devices -- expected 100% | Files heard (/30) |
| F7 | Transcribing speech to text correctly. | 20 spoken phrases -- expected 80% accuracy | Phrases correct (/20) |
| F8 | Sending text input via terminal. | 10 text inputs -- expected 100% | Inputs received from another module (/10) |
| F9 | Receiving text output via terminal. | 10 text outputs from another module -- expected 100% | Outputs displayed (/10) |
| F10 | Switching between the 3 modes (Discussion, Context, Info). | Switch 10 times -- expected 100% success | Switches successful (/10) |
| F11 | Generating audio file from text input. | 10 text inputs -- expected 100% generation | Files generated (/10) |
| F12 | Making human movement | human feeling* -- expected 60% | Human feedback in % |
| F13 | Retrieving context from a saved file. | 10 queries on saved text -- expected 100% retrieval | Relevant text found (/10) |
| F14 | Generating a response from a context. | 10 prompts -- expected 100% answer generation | Answers generated (/10) |
| F15 | Understanding of emotions from the interlocutor. | 10 emotion texts to analyse -- expected 60% | Emotion analysed (/10) |
| F16 | Understanding of the global emotion of the interlocutor | 10 emotional contexts to analyse -- expected 60% | 10 emotional contexts (/10) |
| F17 | Saving of important information with an update that removes information when they are old / not relevant anymore / not remembered | 10 pieces of information to save and treat over time -- expected 60% | 10 prompts to save and treat (/10) |

*Experiments will be conducted on several people and a Godspeed-based questionnaire will evaluate the human feeling