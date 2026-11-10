"""The semi-Lagrangian operator moves fields downstream, by the right amount."""

import math

import torch

from nowcast.advection import deformation, departure_points, estimate_velocity, grid, sample


def blob(h=96, w=96, x0=48.0, y0=48.0, sigma=4.0):
    pos = grid(1, h, w)
    return torch.exp(-((pos[:, :1] - x0) ** 2 + (pos[:, 1:] - y0) ** 2) / (2 * sigma**2))


def centroid(field):
    pos = grid(1, *field.shape[-2:])
    mass = field.sum()
    return (pos[0, 0] * field[0, 0]).sum() / mass, (pos[0, 1] * field[0, 0]).sum() / mass


def test_zero_velocity_is_identity():
    field = blob()
    points = departure_points(torch.zeros(1, 2, 96, 96), steps=6)
    assert torch.allclose(sample(field, points[-1]), field, atol=1e-6)


def test_uniform_translation_moves_field_downstream():
    u, v = 1.5, -0.75  # px per step: +x (east) and -y (south)
    velocity = torch.tensor([u, v]).reshape(1, 2, 1, 1).expand(1, 2, 96, 96)
    field = blob()
    x0, y0 = centroid(field)
    for k, points in enumerate(departure_points(velocity, steps=8), start=1):
        x, y = centroid(sample(field, points))
        assert abs(float(x - x0) - k * u) < 0.05 and abs(float(y - y0) - k * v) < 0.05


def test_solid_body_rotation_follows_circles():
    h = w = 129
    omega = 0.02  # rad per step
    pos = grid(1, h, w)
    rx, ry = pos[:, 0] - 64, pos[:, 1] - 64
    velocity = torch.stack([-omega * ry, omega * rx], dim=1)
    steps = 24
    departure = departure_points(velocity, steps)[-1]
    angle = -omega * steps  # departure point = arrival point rotated backwards
    expected_x = 64 + rx * math.cos(angle) - ry * math.sin(angle)
    expected_y = 64 + rx * math.sin(angle) + ry * math.cos(angle)
    r = torch.sqrt(rx**2 + ry**2)
    within = r < 40  # well inside the domain
    error = torch.sqrt((departure[:, 0] - expected_x) ** 2 + (departure[:, 1] - expected_y) ** 2)
    assert float(error[within].max()) < 0.1  # midpoint rule: O(omega^3) per step


def test_coarse_trajectories_match_fine_ones_for_smooth_flow():
    h = w = 128
    pos = grid(1, h, w)
    fine = torch.stack([1.5 + 0.5 * torch.sin(pos[:, 1] / 20), -1.0 + 0.5 * torch.cos(pos[:, 0] / 25)], dim=1)
    coarse = torch.nn.functional.avg_pool2d(fine, 4)
    reference = departure_points(fine, 12)[-1]
    approx = departure_points(coarse, 12, upsample=4)[-1]
    interior = (slice(None), slice(None), slice(24, -24), slice(24, -24))
    assert float((approx - reference)[interior].abs().max()) < 0.3  # px, after 12 steps (~20 px travelled)


def test_variational_flow_recovers_a_translation():
    torch.manual_seed(0)
    texture = torch.nn.functional.avg_pool2d(torch.rand(1, 1, 160, 160), 9, stride=1, padding=4) * 4
    u, v = 2.0, -1.0  # px per step
    # frame j shows the texture at x + 16 + (u, v) * (5 - j): features move by +(u, v) per step
    frames = torch.cat([sample(texture, grid(1, 128, 128) + torch.tensor([16 + u * (5 - j), 16 + v * (5 - j)]).reshape(1, 2, 1, 1))
                        for j in range(6)], dim=1)
    velocity = estimate_velocity(frames, torch.ones(1, 1, 128, 128, dtype=torch.bool), grid=16, pool=4, iterations=150)
    interior = velocity[..., 2:-2, 2:-2]
    assert abs(float(interior[:, 0].mean()) - u) < 0.2 and abs(float(interior[:, 1].mean()) - v) < 0.2


def test_deformation_of_rigid_and_straining_flows():
    pos = grid(1, 96, 96)
    rx, ry = pos[:, 0] - 48, pos[:, 1] - 48
    rotation = torch.stack([-0.02 * ry, 0.02 * rx], dim=1)  # rigid: no stretching
    strain = torch.stack([0.02 * rx, -0.02 * ry], dim=1)    # pure strain, rate e = 0.02 per step
    centre = (slice(None), slice(30, 66), slice(30, 66))
    anisotropy, det = deformation(departure_points(rotation, 24)[-1])
    assert float((anisotropy[centre] - 1).abs().max()) < 0.01 and float(det[centre].min()) > 0.99
    anisotropy, _ = deformation(departure_points(strain, 24)[-1])
    expected = math.exp(2 * 0.02 * 24)  # material lines stretch like exp(e t) along one axis and shrink along the other
    assert float((anisotropy[centre] / expected - 1).abs().max()) < 0.02


def test_flow_does_not_explain_growth_and_decay_by_strain():
    # A translating texture whose amplitude grows on one side and decays on the other: the fitted
    # motion must stay a translation, so that the 2 h trajectory map does not shear features into streaks.
    torch.manual_seed(0)
    texture = torch.nn.functional.avg_pool2d(torch.rand(1, 1, 200, 200), 9, stride=1, padding=4)
    texture = (texture - texture.mean()) / texture.std() + 1.5  # contrast comparable to the amplitude trend
    u, v = 2.0, -1.0
    trend = 1 + 0.08 * torch.tanh((grid(1, 128, 128)[:, :1] - 64) / 20)  # per step: +8% east, -8% west
    frames = torch.cat([sample(texture, grid(1, 128, 128) + torch.tensor([36 + u * (5 - j), 36 + v * (5 - j)]).reshape(1, 2, 1, 1))
                        * trend ** j for j in range(6)], dim=1)
    velocity = estimate_velocity(frames, torch.ones(1, 1, 128, 128, dtype=torch.bool))
    interior = velocity[..., 2:-2, 2:-2]
    assert abs(float(interior[:, 0].mean()) - u) < 0.25 and abs(float(interior[:, 1].mean()) - v) < 0.25
    anisotropy, det = deformation(departure_points(velocity, 24, upsample=16)[-1])
    assert float(anisotropy.median()) < 1.3 and float(anisotropy.quantile(0.99)) < 2.0 and float(det.min()) > 0.5
    # (with the former smoothness of 0.01: median 1.5, 99th percentile 3.5)
