from minimel import *

eval = evaluate
subcommands = [
    index,
    get_disambig,
    get_paragraphs,
    count,
    count_surface,
    clean,
    vectorize,
    ent_feats,
    train,
    run,
    eval,
    experiment,
    audit,
]


def main():
    import defopt, sys, logging

    _create_parser_old = defopt._create_parser

    def _create_parser_new(*args, **kwargs):
        global client

        parser = _create_parser_old(*args, **kwargs)
        parser.add_argument(
            "--verbose",
            "-v",
            action="count",
            default=0,
            help="Verbosity (use -vv for debug messages)",
        )
        parser.add_argument("--slurm", "-s", action="store_true", help="Use Slurm")

        args = parser.parse_args(sys.argv[1:])
        logging.basicConfig(level=30 - (args.verbose * 10))

        from dask.distributed import Client, LocalCluster, TimeoutError, CancelledError

        if args.slurm:
            from dask_jobqueue import SLURMCluster

            try:
                client = Client('tcp://localhost:8883', timeout='5s')
            except (TimeoutError, OSError):
                logging.info("Setting up SLURM cluster...")
                cluster = SLURMCluster(
                    cores=4,
                    processes=4,
                    memory="64GB",
                    project="minimel",
                    walltime="00:15:00",
                    dashboard_address=":8883",
                )
                cluster.scale(jobs=16)
                client = Client(cluster)
            logging.info(f"Running on {client}")
        # else:
        #     cluster = LocalCluster(n_workers=4, threads_per_worker=2)
        #     client = Client(cluster)
        #     logging.info(f"Running on {client}")

        return parser

    defopt._create_parser = _create_parser_new

    defopt.run(subcommands)


if __name__ == "__main__":
    main()
