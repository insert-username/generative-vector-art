#!/usr/bin/env python3

from main import *

def generate_suite(ROWS, COLUMNS):

    SVG_WIDTH=400
    SVG_HEIGHT=400
    BORDER_SIZE=20
    CELL_LINE_WIDTH=(SVG_WIDTH/ROWS) / 3 

    # Standard
    maze = Maze(ROWS, COLUMNS)
    bias_provider = ConstDirectionalBiasProvider(0.5)
    neighbor_selector = DirectionalBiasedNeighborSelector(bias_provider)
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"standard{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Horizontally biased
    maze = Maze(ROWS, COLUMNS)
    bias_provider = ConstDirectionalBiasProvider(0.8)
    neighbor_selector = DirectionalBiasedNeighborSelector(bias_provider)
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"horiz-bias-08-{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    maze = Maze(ROWS, COLUMNS)
    bias_provider = ConstDirectionalBiasProvider(0.6)
    neighbor_selector = DirectionalBiasedNeighborSelector(bias_provider)
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"horiz-bias-06-{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Curling
    maze = Maze(ROWS, COLUMNS)
    neighbor_selector = CurlingNS(5)
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"curling{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Image Circle
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/circle.png")
    bias_provider = ImageDirectionalBWBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image-circ{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Image Stripe
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/stripe.png")
    bias_provider = ImageDirectionalBWBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image-stripe{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Image Lightning
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/lightning.png")
    bias_provider = ImageDirectionalBWBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image-lightning{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    #Gradient
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/grad.png")
    bias_provider = ImageDirectionalBWBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image-grad{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)


    #rotational biased images


    # Image Circle
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/circle.png")
    bias_provider = ImageDirectionalBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image1-circ{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Image Stripe
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/stripe.png")
    bias_provider = ImageDirectionalBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image1-stripe{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    # Image Lightning
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/lightning.png")
    bias_provider = ImageDirectionalBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image1-lightning{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

    #Gradient
    maze = Maze(ROWS, COLUMNS)
    image = Image.open("./images/grad.png")
    bias_provider = ImageDirectionalBiasProvider(image, maze)
    neighbor_selector = ImageDelegatingNS(image, maze, \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.2)), \
            DirectionalBiasedNeighborSelector(ConstDirectionalBiasProvider(0.8)))
    gen = Generator(maze, neighbor_selector)
    gen.run_all_steps()
    create_coaster(maze, args.output_dir + f"image1-grad{ROWS}", SVG_WIDTH, SVG_HEIGHT, BORDER_SIZE, CELL_LINE_WIDTH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a suite of coasters.")
    parser.add_argument("output_dir", metavar="OUTPUT_DIR", type=str)
    args = parser.parse_args()

    generate_suite(32, 32)
    generate_suite(64, 64)
