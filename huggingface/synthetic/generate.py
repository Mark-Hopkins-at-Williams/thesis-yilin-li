import random

with open('nounlist.txt') as reader:
    nouns = [line.strip() for line in reader]
with open('verbs.txt') as reader:
    verbs = [line.strip() for line in reader]
with open('intransitive.txt') as reader:
    intransitives = [line.strip() for line in reader]


for i in range(50000):
    print("You {} the {}".format(random.choice(verbs), random.choice(nouns)))
    print("You {}".format(random.choice(intransitives)))
