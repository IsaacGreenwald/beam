from crag.thermal import ThermalParams, net_flux, sky_temperature

PARAMS = ThermalParams(thermal_mass=2.0e5, absorptivity=0.6)

# a vertical open face: sees half the sky
SKY_VIEW = 0.5


def _night_flux(rock_temp_c, cloud, temp_air=10.0):
    return net_flux(
        rock_temp_c,
        dni=0.0,
        diffuse=0.0,
        temp_air=temp_air,
        wind=1.0,
        cloud=cloud,
        beam_factor=0.0,
        sky_view=SKY_VIEW,
        params=PARAMS,
    )


def test_sky_is_colder_than_air_when_clear():
    assert sky_temperature(15.0, cloud_pct=0) < 15.0
    # full cloud pulls the sky back up toward air temperature
    assert sky_temperature(15.0, cloud_pct=100) > sky_temperature(15.0, cloud_pct=0)


def test_sun_adds_energy():
    day = net_flux(
        20.0, dni=800.0, diffuse=120.0, temp_air=20.0, wind=1.0, cloud=0.0,
        beam_factor=0.9, sky_view=SKY_VIEW, params=PARAMS,
    )
    shade = net_flux(
        20.0, dni=800.0, diffuse=120.0, temp_air=20.0, wind=1.0, cloud=0.0,
        beam_factor=0.0, sky_view=SKY_VIEW, params=PARAMS,
    )
    assert day > shade


def test_clear_night_cools_more_than_cloudy():
    # rock warmer than air at night -> both lose heat, but clear sky loses more
    clear = _night_flux(rock_temp_c=12.0, cloud=0.0)
    cloudy = _night_flux(rock_temp_c=12.0, cloud=100.0)
    assert clear < cloudy < 0


def test_convection_pulls_toward_air():
    hot_rock = _night_flux(rock_temp_c=25.0, cloud=100.0, temp_air=10.0)
    cold_rock = _night_flux(rock_temp_c=-5.0, cloud=100.0, temp_air=10.0)
    assert hot_rock < 0  # hot rock in cool air -> loses heat
    assert cold_rock > 0  # cold rock in warmer air -> gains heat
