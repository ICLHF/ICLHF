from environment.simulator import Simulator


def main(config_path: str):
    sim = Simulator(config_path)
    sim.start()
    # sim.run()


if __name__ == "__main__":
    main("config/scene/messy_table.yml")
