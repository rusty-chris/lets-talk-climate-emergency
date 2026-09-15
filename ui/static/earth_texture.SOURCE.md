earth_texture.jpg — provenance & licence
=========================================

What:   Equirectangular (2:1) Earth surface texture (land, shallow water,
        shaded topography), downscaled to 1024x512 and re-encoded as JPEG
        (quality 72) for use as a baked data-URI in ui/theme.py (the hero
        planet and the spinning-globe loader).

Source: NASA Visible Earth — "Blue Marble: Land Surface, Shallow Water, and
        Shaded Topography" (Reto Stöckli, NASA Earth Observatory / NASA
        Goddard Space Flight Center). Retrieved from eoimages.gsfc.nasa.gov
        (a NASA .gov host) on 2026-09-15.

Licence: Public domain. NASA imagery is a work of the U.S. federal government
        and is not subject to copyright (see NASA media-usage guidelines).
        No attribution is legally required; the credit above is courtesy.

Why baked, not fetched: the product follows a strict no-external-requests
        convention (CSP; transparency pages). The image is committed to the
        repo and inlined as a data-URI at import — nothing is fetched at
        runtime. The NASA source URL is intentionally NOT placed in ui/theme.py
        so the module's "no external URL" test guard holds; it lives here.
