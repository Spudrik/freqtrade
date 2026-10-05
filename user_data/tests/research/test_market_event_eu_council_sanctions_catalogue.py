# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
import pytest


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_eu_council_sanctions_catalogue"
)
eu = importlib.import_module(MODULE)


def page_html(entries: list[tuple[str, str, str, str]], *, last_page: int = 1) -> bytes:
    cards = []
    for path, title, clock, summary in entries:
        cards.append(
            f"""
            <li class="gsc-excerpt-item"><a class="gsc-excerpt-item__link" href="{path}">
              <span class="gsc-excerpt-item__title">{title}</span>
              <time class="gsc-time-badge" datetime="2/22/2021 {clock}:00 AM">{clock}</time>
              <div id="excerpt-text"><p>{summary}</p></div>
              <span id="excerpt-tag">Council of the EU</span>
            </a></li>
            """
        )
    pages = "".join(
        f'<a href="{eu.ARCHIVE_PATH}?dateFrom={eu.START_DATE}&dateTo={eu.END_DATE}'
        f'&topic={eu.SANCTIONS_TOPIC_ID}&page={page}">{page}</a>'
        for page in range(1, last_page + 1)
    )
    return (
        "<html><body><h1>Press releases and statements</h1>"
        '<ul><li class="gsc-excerpt-list__item">'
        '<h2 class="gsc-excerpt-list__item-date">22 February 2021</h2><ul>'
        + "".join(cards)
        + f'</ul></li></ul><nav aria-label="Select a page">{pages}</nav></body></html>'
    ).encode()


def test_classification_keeps_actions_and_excludes_administration() -> None:
    assert eu.classify_release(
        "Venezuela: 19 officials added to the EU sanctions list", ""
    ) == ("include", "substantive_tightening_or_new_listing")
    assert eu.classify_release(
        "Russia: EU adopts new sanctions framework", "Restrictive measures"
    ) == ("include", "new_sanctions_framework_or_regime")
    assert eu.classify_release(
        "EU lifts sanctions on one person", "Council removes one listing"
    ) == ("include", "substantive_easing_or_removal")
    assert eu.classify_release(
        "Statement on alignment of certain third countries concerning sanctions", ""
    ) == ("exclude", "third_country_alignment_notice")
    assert eu.classify_release(
        "EU renews restrictive measures for another year", "No new listings"
    ) == ("exclude", "routine_renewal_or_extension")


@pytest.mark.parametrize(
    ("title", "summary", "expected"),
    [
        (
            "Sanctions targeting terrorism: Council renews EU terrorist list",
            "The list contains individuals and entities subject to sanctions.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "EU individual sanctions prolonged for a further six months",
            "Restrictive measures against individuals and entities were renewed.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "Sanctions against terrorism: Council renews the EU Terrorist List "
            "and designates a new entity",
            "The Council added one entity to the sanctions list.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Venezuela: Council renews restrictive measures and lists a further "
            "15 individuals",
            "The Council decided to list 15 more individuals.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Zimbabwe: Council renews restrictive measures framework and delists "
            "one entity",
            "The Council delisted Zimbabwe Defence Industries.",
            ("include", "substantive_easing_or_removal"),
        ),
        (
            "Russia: Council removes six listings from sanctions framework",
            "Six individuals were removed from the sanctions list.",
            ("include", "substantive_easing_or_removal"),
        ),
    ],
)
def test_classification_distinguishes_renewals_from_explicit_deltas(
    title: str, summary: str, expected: tuple[str, str]
) -> None:
    assert eu.classify_release(title, summary) == expected


@pytest.mark.parametrize(
    ("title", "summary", "expected"),
    [
        (
            "Humanitarian action: EU introduces exemptions to sanctions",
            "The Council modified restrictive measures to introduce exemptions.",
            ("include", "substantive_easing_or_removal"),
        ),
        (
            "Wagner Group and RIA FAN added to the EU's sanctions list",
            "The Council imposed restrictive measures on both entities.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Iran: Council broadens EU restrictive measures",
            "The Council broadened the scope of the sanctions framework.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Syria: Council statement on the lifting of EU economic sanctions",
            "The Council issued a statement on the lifting of sanctions.",
            ("include", "substantive_easing_or_removal"),
        ),
        (
            "Democratic Republic of the Congo: Statement by the High Representative",
            "The EU welcomed a UN report mandated by the UN Sanctions Committee.",
            ("exclude", "statement_or_conclusions_without_new_action"),
        ),
        (
            "Humanitarian exemption: Council extends existing exemption for a year",
            "The Council extended the humanitarian exemption to the sanctions regime, "
            "introduced after an earthquake.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "EU extends sanctions for a further year",
            "The Council extended sanctions introduced by the EU in response to an "
            "earlier event.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "Iran sanctions snapback: Council reimposes restrictive measures",
            "The Council reimposed measures that had previously been lifted.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Haiti: EU sanctions three individuals",
            "The Council adopted restrictive measures against three individuals.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "EU adopts its 16th package of restrictive measures",
            "The package targets sectors of the economy.",
            ("include", "substantive_tightening_or_new_listing"),
        ),
        (
            "Sudan: EU sanctions regime prolonged for a further year",
            "The Council extended restrictive measures for a further year.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "Chemical Weapons: EU sanctions renewed for an additional year",
            "The Council prolonged restrictive measures for one year.",
            ("exclude", "routine_renewal_or_extension"),
        ),
        (
            "Humanitarian action: EU introduces further exceptions to sanctions to "
            "facilitate the delivery of assistance",
            "The Council introduced humanitarian exceptions to asset freeze measures "
            "in 10 EU restrictive measures regimes.",
            ("include", "substantive_easing_or_removal"),
        ),
        (
            "Humanitarian action: EU introduces further exception to sanctions",
            "The Council introduced a humanitarian exception to asset freeze measures.",
            ("include", "substantive_easing_or_removal"),
        ),
        (
            "Haiti: EU sets up autonomous framework for restrictive measures",
            "The Council amended its sanctions regime to allow restrictive measures.",
            ("include", "new_sanctions_framework_or_regime"),
        ),
        (
            "Russia: EU sets up new country-specific framework for restrictive measures",
            "The Council established a new sanctions regime and listed 20 persons.",
            ("include", "new_sanctions_framework_or_regime"),
        ),
    ],
)
def test_classification_resolves_frozen_manual_language(
    title: str, summary: str, expected: tuple[str, str]
) -> None:
    assert eu.classify_release(title, summary) == expected


def test_parse_and_build_are_outcome_blind_and_episode_safe() -> None:
    html = page_html(
        [
            (
                "/en/press/press-releases/2021/02/22/one/",
                "Venezuela: 19 officials added to the EU sanctions list",
                "11:48",
                "The Council added additional officials.",
            ),
            (
                "/en/press/press-releases/2021/02/22/two/",
                "EU imposes further restrictive measures on ten people",
                "11:58",
                "The Council sanctioned ten additional people.",
            ),
        ]
    )
    frame = eu.build_catalogue([html])
    assert len(frame) == 2
    assert frame["selection_status"].eq("include").all()
    assert frame["episode_id"].nunique() == 1
    assert frame["episode_members"].eq(2).all()
    assert pd.to_datetime(frame["displayed_publication_clock"], utc=True).dt.tz is not None
    forbidden = ("profit", "future_return", "price", "volume", "direction")
    assert not any(token in column for column in frame.columns for token in forbidden)
    freeze = eu.freeze_document(frame, eu.count_catalogue(frame))
    assert any(
        "not a complete historical universe of all EU sanctions before 2023" in boundary
        for boundary in freeze["boundaries"]
    )


def test_page_count_and_response_url_are_frozen() -> None:
    html = page_html(
        [
            (
                "/en/press/press-releases/2021/02/22/one/",
                "EU adds new sanctions listings",
                "11:48",
                "Additional restrictive measures.",
            )
        ],
        last_page=18,
    )
    assert eu.total_pages(html) == 18
    assert eu.valid_archive_response_url(eu.archive_url(18), page=18)
    assert not eu.valid_archive_response_url(
        eu.archive_url(18).replace(eu.MIRROR_HOST, "example.com"), page=18
    )
