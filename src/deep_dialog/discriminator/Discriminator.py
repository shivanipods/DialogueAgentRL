## collect N simulated dialogues
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_path', dest='expert_path', type=str,
                        default=None,
                        help="Path to human-human training data corpus.")
    parser.add_argument('--expert_type', dest='expert_type', type=str, default='dqn', help="type of gold dialogues for discriminator")
    parser.add_argument('--mode', dest='mode', type=str, default="train")


def run_expert_agent():
    ## run the dqn agent for a long time
    ## collect 50 succesful dialogues with the lowest number of turns
    ## collect 50 turns which are very bad
    ## storage format
    return


def main():
    args = parse_arguments()

    if args.mode == "generate_data":
        run_expert_agent()

if __name__ == '__main__':
    main()