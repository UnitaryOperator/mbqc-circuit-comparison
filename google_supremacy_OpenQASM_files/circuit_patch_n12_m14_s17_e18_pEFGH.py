import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(3, 3),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(5, 6),
]

CIRCUIT = cirq.Circuit(
    [
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.040290133403821).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.2704822022121594).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.8693959871027745).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.5809728937821896).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.15501230573908442).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.07517976306925434).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.1118243782923773).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.4002474716129623).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.4271137305918131).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.5726071620504323).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.454015769694289).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.130087599403273).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.5101889326253684).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.6521267190396145).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.697283830215655).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.5738562558524603).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.055658030068537).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.9101645986099156).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.4085062699097115).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.0845780996186942).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.9901684598272027).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=3.132106246241449).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.2999942968082916).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.42342187117148633).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.2660532793737076).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.076572502053615).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.9139968813391837).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.8307414448788641).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.1550137375727765).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.2809287459890495).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.9609784884872923).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.7714977111672).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.4656516464003069).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.3823962099399854).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.818060579169371).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.029182244448389).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.210418068606561).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.472316890149653).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.6831000466013748).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.39710795546356087).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.4273196554466487).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.423156517170531).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.499325783594923).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.2374269620518312).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.602640638400075).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.888632729537889).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.4276604187772115).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.42349728050109187).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.2970548008672864).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.0668627320589472).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.6366214143036792).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.9250445076242624).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.1823326285320235).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.4125246973403627).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.3941930231140738).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.1057699297934924).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-3.07890367081464).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.9334102393560215).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-3.1380191536956925).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.821237983192878).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.247155881068256).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.1052180946540027).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.4962219003856738).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.37279432602247553).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.4503593713379175).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.5958528027965357).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.7173558861201048).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.39342771582908925).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.46432796634124074).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.606265752755494).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.09893236697830686).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.22235994134150514).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.5111146263998378).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.7005954037199302).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.949188546417883).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=3.0324439828782097).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-3.054720418237576).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.7925224053801756).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.5450389129187485).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.7345196902388409).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.0456517669778016).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.03760366948252525).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.3916735766409865).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.04426890692085195).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.0741350641658407).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.33603388570893244).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.1641564337096924).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.4501485248475063).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.31616939223411933).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.31200625395799975).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.6369572208457974).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.8988560423888892).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.24461584191099206).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.041376249226818373).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.538810681989734).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.5346475437136284).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.6487855720411915).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.878977640849527).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.1405464914694612).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.14787660185112372).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.7635077443764546).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.5333156755681188).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.382974882659063).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.6713979759796551).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.30173576504151).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.15624233358288464).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.446868769906086).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.770796940197105).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.2786846124176954).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.42062239883194863).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.2951599705556909).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.17173239619247838).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.3268085344352123).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.181315102976587).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.026205502330501677).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.2977226679605174).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.0615125271447248).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.9195747407304715).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.10212956285167607).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.02129801151153643).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.9949027750062065).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.8054219976861).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.5291886669953776).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.6124441034557042).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.9812692668683418).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.41721175043018).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.2321289928548005).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.042648215534694).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.3743481124447179).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.4576035489050305).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.6817775747282475).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.117720058290086).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.0621479402748655).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.800249118731774).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=3.011412914020756).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.985780302021009).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.7949808709784174).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.7991440092545368).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.7732402252865036).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-3.0351390468295953).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.0918723222220628).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.8058802310842417).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.633224361977309).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.6373875002534284).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.3114406377700867).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.5416327065784223).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.9177143972425874).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.629291303922006).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.5738371898946504).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.804029258702986).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.1230425187473863).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.834619425426805).on(cirq.GridQubit(3, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.4754321407316304).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.6209255721902416).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.7557183861164773).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.0796465564074964).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.804525105903661).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.946462892317914).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.09409804072570793).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.02932953363750457).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.1792088669712477).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.324702298429859).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.6649448814591068).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.9888730517501259).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.695832286548896).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.8377700729631492).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.303191492681659).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.17976391831844651).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
    ]
)

