"""Evidence-based rules for assessing a used laptop listing."""
from dataclasses import dataclass
from statistics import median

CONDITIONS = ("Unknown or untested", "Working, normal wear", "Working, battery worn", "Fault or damage")
WORKLOADS = ("Everyday work and study", "Development and heavy multitasking",
             "Modern gaming", "Video editing and 3D")


@dataclass(frozen=True)
class Assessment:
    verdict: str
    reason: str
    next_step: str


@dataclass(frozen=True)
class HardwareFit:
    title: str
    reason: str
    suitable: bool | None


def market_assessment(ask, comparisons, source):
    if ask is None:
        return Assessment("Enter the seller’s asking price",
                          "An asking price lets you compare the offer with your evidence.",
                          "Enter the total price per laptop, including any required fees.")
    if len(comparisons) < 3:
        return Assessment("Used-market value needs more evidence",
                          f"You entered {len(comparisons)} of the 3 comparable prices needed for a verdict.",
                          "Find recent prices for the same model, specs, and condition in your market. "
                          "Prefer completed sales where you can find them.")

    low, middle, high = min(comparisons), median(comparisons), max(comparisons)
    evidence = (f"The {len(comparisons)} {source.lower()} you entered range from €{low:,.2f} to €{high:,.2f}. "
                f"The median is €{middle:,.2f}.")
    gap = round(ask - middle, 2)
    if gap:
        direction = "above" if gap > 0 else "below"
        evidence += f" The asking price is €{abs(gap):,.2f} {direction} that median."
    if source == "Asking prices":
        evidence += " These are seller quotes and may differ from what buyers pay."
    if ask > high:
        return Assessment("Expensive against your comparisons", evidence,
                          f"Ask the seller to justify the premium or negotiate toward €{middle:,.2f}. "
                          "A verified warranty or better condition could justify paying more.")
    if ask < low:
        return Assessment("Potentially good value", evidence,
                          "Shortlist it if it fits your workload. Check why it is cheaper and verify its "
                          "condition and specs before paying.")
    if ask > middle:
        return Assessment("Reasonable, toward the higher end", evidence,
                          f"Try to negotiate toward €{middle:,.2f}, especially if a cheaper comparable unit "
                          "has similar condition and warranty.")
    return Assessment("Competitive against your comparisons", evidence,
                      "Shortlist it if it fits your workload. Compare condition, return terms and "
                      "accessories with the other offers before choosing.")


def hardware_fit(row, workload):
    ram = row["Ram"]
    ssd = row["SSD_GB"]
    gpu = row["Gpu_Model"]
    if workload == WORKLOADS[0]:
        if ram < 8:
            return HardwareFit("Limited for everyday multitasking",
                f"{ram:g} GB of RAM leaves little room for several browser tabs and a video call. "
                "Test your normal workload or check whether the memory can be upgraded.", False,
            )
        storage = ("Solid-state storage helps with startup and opening apps." if ssd else
                   "Hard-drive-only storage can make startup and app loading feel slow.")
        return HardwareFit("Suitable on paper for everyday work",
            f"{ram:g} GB of RAM is a reasonable starting point for browsing, documents, and light "
            f"multitasking. {storage} Check that your required operating system and apps are supported.", True,
        )

    if workload == WORKLOADS[1]:
        if ram < 16 or ssd < 256:
            return HardwareFit("Limited for heavy multitasking",
                f"With {ram:g} GB of RAM and {ssd:g} GB of solid-state storage, large projects, containers "
                "or virtual machines could be cramped. Check memory and storage upgrade options.", False,
            )
        return HardwareFit("Worth testing for development",
            f"{ram:g} GB of RAM and {ssd:g} GB of solid-state storage are a good starting point. Run a "
            "representative build or development workload to check CPU performance and thermals.", None,
        )

    if row["Gpu_Brand"] == "Intel" or ram < 16:
        title = "Weak candidate for demanding games" if workload == WORKLOADS[2] else "Limited for demanding editing or 3D"
        return HardwareFit(title,
            f"This build has {gpu} graphics and {ram:g} GB of RAM. Check the exact GPU, memory, and "
            "driver requirements of the games or applications you plan to use.", False,
        )
    target = "games, resolution, and settings" if workload == WORKLOADS[2] else "editor, codecs, and project size"
    return HardwareFit("Test this configuration before buying",
        f"It has {gpu} graphics and {ram:g} GB of RAM. Check the GPU’s supported features and driver "
        f"requirements, then run a representative test for your {target}.", None,
    )


def condition_check(condition):
    if condition == CONDITIONS[0]:
        return "Hold off until it passes inspection", (
            "Ask for a boot test, battery health report, and proof that the listed specs match. "
            "Verify the charger, screen, ports, and return terms before paying."
        )
    if condition == CONDITIONS[2]:
        return "Factor in the worn battery", (
            "Get a replacement quote and compare the total cost with similar working units. "
            "Ask the seller for a battery report and check charger and return terms."
        )
    if condition == CONDITIONS[3]:
        return "Pass for ordinary use until the fault is resolved", (
            "Get a repair diagnosis and quote before committing. If you are buying for parts, "
            "compare it with listings that have the same fault."
        )
    return "Verify the seller’s condition claim", (
        "Check the battery report, screen, keyboard, ports, charger, and return terms. "
        "For a batch purchase, inspect a sample unit before committing."
    )


def purchase_guidance(condition, fit):
    title, checks = condition_check(condition)
    if condition == CONDITIONS[3]:
        return title, checks
    if fit.suitable is False:
        return "Look for a better fit for your workload", (
            "The selected workload is likely to be constrained by this configuration. "
            "Shortlist a better-equipped laptop even if this quote looks competitive."
        )
    if fit.suitable is None and condition == CONDITIONS[1]:
        return "Test your workload before committing", checks
    return title, checks
