'''test methods for the OrbitalBallRoller class in the rolling ball package
'''

from nose.tools import raises
from RollingBall import rolling_ball


class TestBallRollerInstantiation():
    '''test the orbital ball roller object instantiation
    '''
    def test_ball_roller_object(self):
        '''test successful instantiation of an OrbitalBallRoller object
        '''
        points = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        rolling_ball.OrbitalBallRoller(points)

    @raises(ValueError)
    def test_short_input_array(self):
        '''test successful instantiation of an OrbitalBallRoller object
        '''
        points = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        rolling_ball.OrbitalBallRoller(points)


class TestBallRollerMethods():
    '''test basic functionality of ball roller object
    '''
    def setup(self):
        '''setup function to be called before each test
        '''
        # gots 2 load in some test data here
        points = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        self.ball_roller = rolling_ball.OrbitalBallRoller(points)

    def teardown(self):
        '''teardown function to be called after each test
        '''
        del self.ball_roller

    def test_get_alpha_complex(self):
        '''test the creation of an alpha complex from the triangulation
        '''
        self.ball_roller.get_alpha_complex(alpha=.5)

    def test_plot_delaunay(self):
        '''test the plot routine of the triangulation
        '''
        self.ball_roller.plot_delaunay()

    def test_get_background(self):
        '''test the routine that acquires the background density
        '''
        self.ball_roller.get_background()

    @raises()
    def test_get_background_exception(self):
        '''test the the routine to acquire the background density throws error
           if no alpha_complex has been determined for the object
        '''
        self.ball_roller.get_background()

    def test_locate_depletions(self):
        '''test the method that identifies plasma depletions from the
           background density
        '''
        self.ball_roller.locate_depletions()

    @raises()
    def test_locate_depletions_exception(self):
        '''test that the locate depletions method produces exception if ...
        '''
        self.ball_roller.locate_depletions()
