import launcher
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # Launch the desired bot with the parameters defined in the configuration.yml file
    bot = launcher.bot()
    logging.info(bot)


if __name__ == "__main__":
    main()