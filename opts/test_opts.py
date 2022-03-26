from .base_opts import BaseOptions

class TestOptions(BaseOptions):
    """This class includes testing options.
    
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--test_checkpoints', type=str, help='use which models to inference. split by semicolon.')
        parser.add_argument('--test_log_dir', type=str, default='./test_logs', help='test logs are saved here')
        parser.add_argument('--test_results', type=str, help='use which results to do ensemble. split by semicolon.')
        parser.add_argument('--submit_dir', type=str, default='submit', help='submit save dir')
        parser.add_argument('--smoothed', type=str, default='n', choices=['y', 'n'], help='whether to use the smoothed prediction sequence when do ensemble and generating the final txt files')
        parser.add_argument('--test_target', type=str, default='None', help='arousal, valence')
        self.isTrain = False
        return parser