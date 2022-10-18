import argparse

def test_parse(opts):
    print("#########################")
    print("Testing here:")
    print("the first arg is: ", opts.bne_gp_lengthscale, ", and the second is: ", opts.bma_gp_l2_regularizer)
    print("Thanks for testing!")
    print("#########################")


if __name__ == '__main__':
    opts = argparse.ArgumentParser(
        description="Launch a batch of experiments"
    )

    opts.add_argument(
        "--bne_gp_lengthscale",
        help="the first parameter to tune",
        type=float,
        default=1,
    )
    opts.add_argument(
        "--bma_gp_l2_regularizer",
        help="the second parameter to tune",
        type=float,
        default=1e-3,
    )
    opts = opts.parse_args()

    test_parse(opts)
