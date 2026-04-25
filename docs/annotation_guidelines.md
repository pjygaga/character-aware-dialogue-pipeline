What you need to do
you are going to read a line said by Hermione Granger from Harry Potter movies. your job is put one emotion label on it. only one label per line.

the six labels you can choose from:

determined
worried
angry
happy
sad
neutral

do not invent new labels. if you're not sure, just pick the closest one.

Core rules:
work line by line. dont try to infer context from other lines.
if you see a line and you have no idea, go with neutral.
be consistant. if you labeled one type of line determined the first time, do the same thing the 500th time.
dont overthink it. 3 seconds per line is fine.

definition for each label

determined
this is when Hermione is confident, pushing forward, making a plan, correcting someone, or explaining something she knows. a LOT of her lines fall into this. she's a know-it-all so this should be common.
example:

"It's leviOsa, not levioSAH."
"We can't be seen, Harry."
"I read about this in Hogwarts, A History."

if she sounds like she's in charge or teaching, its determined.


worried
she is scared, anxious, nervous, or thinking about something bad that might happen. future-oriented fear usually. she mentions a problem or a risk.
example:

"What if they catch us?"
"This isn't good. This really isn't good."
"I don't like this, Harry."

worried vs sad is tricky. worried is about something that might happen. sad is about something that already happened.


angry
she is mad, annoyed, yelling at someone, or snapping. usually short and sharp. often directed at ron.
exampels:

"Ronald!"
"How dare you!"
"That is the most ridiculous thing I've ever heard."

happy
she is glad, laughing, celebrating, relieved, or complimenting someone. not super common actually, hermione isn't a super happy character. but sometimes.
example:

"Oh Harry, we did it!"
"That was brilliant, Ron."

sad
something bad already happened. she is crying, mourning, disappointed, or feeling hopeless. PAST oriented.
examples:

"I can't believe he's gone."
"I just... I don't know what to do anymore."

### neutral (USE SPARINGLY)

CRITICAL: Do NOT use neutral as a default or safe choice. Most Hermione lines have some emotional undertone.

ONLY use neutral if ALL of these are true:
- The line is a pure factual statement with no emotional coloring
- No urgency, no worry, no assertion, no frustration
- It would sound identical if said by any character

Examples of TRUE neutral (rare):
- "Yes." (as acknowledgment)
- "It's on the third floor."

Examples that are NOT neutral (common mistakes):
- "We need to go now" -> determined (urgency)
- "Harry, listen" -> determined (assertion)
- "I don't think so" -> worried or determined depending on context

If you're choosing between neutral and ANY other label, choose the other label.

priority rules (when two emotions could fit)
sometimes a line can feel like two emotions at the same time. here are the rules:

determined beats worried. if she's worried but still giving a plan, its determined. example: "We need to find it before they do" sounds worried but its a plan, so -> determined.
angry beats sad. if she's crying AND yelling, pick angry. because angry is more action-y.
neutral is the last resort. don't pick neutral just because you're lazy. only pick it when really nothing else fits.
if its still a tie, pick the one that would be more useful for image generation. because that's the downstream task. so determined/angry/worried are more useful than happy/sad/neutral (they have stronger visual signals).
angry is about other people or things that make her frustrated (e.g., Ron being lazy, Malfoy being cruel). worried is about threats or bad outcomes (e.g., getting caught, failing exams). If the line is about a person -> probably angry. If the line is about a future event -> probably worried


things to watch for:
sometimes Hermione corrects someone (like ron). thats NOT angry, thats determined. she's not mad, she's teaching.
rhetorical questions can be angry OR worried depending on tone. "How could you?" is angry. "What are we going to do?" is worried.
if the line is super short like "Honestly!" use the context clue in the line itself. "Honestly!" on its own -> angry (she's frustrated). but you only get the line, no context, so just go with the tone.
dont confuse excited with happy. excited about a plan -> determined. excited because something good happened -> happy.


output format
just output the label. lowercase, no punctuation, no explanation.
like this:
determined
not like this:
The emotion is: Determined.
and not this either:
determined (because she's giving a plan)
just the word. one word. lowercase.

examples for calibration
read these 10 examples to get the feeling before you start:

"I read about it in Hogwarts, a History." -> determined
"Harry, we shouldn't be here." -> worried
"You're a prat, Ronald Weasley." -> angry
"Oh Harry, I was so worried about you." -> happy (she's relieved, which is happy-adjacent)
"He's gone, Harry. He's really gone." -> sad
"It's on the third floor." -> neutral
"We have to go NOW." -> determined (urgency + action plan)
"What if it doesn't work?" -> worried
"How could you do this to me?" -> angry
"Yes, Professor." -> neutral

if you disagree with any of these you're probably calibrated wrong. re-read the definitions.
