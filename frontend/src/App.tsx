import React, { useMemo, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  MapPin,
  Github,
  Thermometer,
  Factory,
  Droplets,
  Trees,
  Users,
  Cloud,
} from "lucide-react";
import {
  ResponsiveContainer,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { MapContainer, Rectangle, Circle, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";

export type RegionData = {
  lat: number;
  lon: number;
  air_pm25: number;
  air_no2: number;
  water_turbidity: number;
  ndvi: number;
  pop_density: number;
  lst_c: number;
  industrial_km: number;
};

const DEFAULT_POINT = { lat: 19.076, lon: 72.8777 };
const DEFAULT_REGION: RegionData = {
  lat: DEFAULT_POINT.lat,
  lon: DEFAULT_POINT.lon,
  air_pm25: 40,
  air_no2: 25,
  water_turbidity: 5,
  ndvi: 0.4,
  pop_density: 12000,
  lst_c: 32,
  industrial_km: 3,
};

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function isValidRegionData(d: any): d is RegionData {
  return (
    d &&
    [
      d.lat,
      d.lon,
      d.air_pm25,
      d.air_no2,
      d.water_turbidity,
      d.ndvi,
      d.pop_density,
      d.lst_c,
      d.industrial_km,
    ].every(isFiniteNumber)
  );
}

function noise(lat: number, lon: number, seed: number) {
  return Math.abs(Math.sin(lat * 5 + lon * 3 + seed)) % 1.0;
}

function fetchRegionData(lat: number, lon: number): RegionData {
  const n1 = noise(lat, lon, 1);
  const n2 = noise(lat, lon, 2);
  const n3 = noise(lat, lon, 3);
  const n4 = noise(lat, lon, 4);
  const n5 = noise(lat, lon, 5);
  const n6 = noise(lat, lon, 6);
  const n7 = noise(lat, lon, 7);

  const candidate: RegionData = {
    lat,
    lon,
    air_pm25: +(20 + 80 * n1).toFixed(1),
    air_no2: +(10 + 60 * n2).toFixed(1),
    water_turbidity: +(1 + 15 * n3).toFixed(1),
    ndvi: +(0.15 + 0.6 * n4).toFixed(2),
    pop_density: Math.round(3000 + 25000 * n5),
    lst_c: +(28 + 10 * n6).toFixed(1),
    industrial_km: +(0.1 + 8.0 * n7).toFixed(2),
  };
  return isValidRegionData(candidate) ? candidate : DEFAULT_REGION;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function minmax(v: number, lo: number, hi: number, invert = false) {
  const x = clamp(v, lo, hi);
  const score = (x - lo) / (hi - lo || 1);
  return invert ? 1 - score : score;
}

function computeScores(d: RegionData) {
  const safe = isValidRegionData(d) ? d : DEFAULT_REGION;
  return {
    air:
      0.5 * minmax(safe.air_pm25, 10, 100, true) +
      0.5 * minmax(safe.air_no2, 5, 80, true),
    water: minmax(safe.water_turbidity, 1, 20, true),
    green: minmax(safe.ndvi, 0.1, 0.8, false),
    population: minmax(safe.pop_density, 1000, 30000, true),
    temperature: minmax(safe.lst_c, 26, 40, true),
    industrial: minmax(safe.industrial_km, 0.1, 10.0, false),
  } as const;
}

function composite(scores: ReturnType<typeof computeScores>) {
  const w = {
    air: 0.22,
    water: 0.14,
    green: 0.18,
    population: 0.12,
    temperature: 0.2,
    industrial: 0.14,
  };
  const val =
    scores.air * w.air +
    scores.water * w.water +
    scores.green * w.green +
    scores.population * w.population +
    scores.temperature * w.temperature +
    scores.industrial * w.industrial;
  return +val.toFixed(3);
}

function recommendations(
  d: RegionData,
  s: ReturnType<typeof computeScores>
) {
  const safe = isValidRegionData(d) ? d : DEFAULT_REGION;
  const hci = composite(s);
  const habitability =
    hci >= 0.7
      ? "Generally habitable – favorable profile."
      : hci >= 0.5
      ? "Marginally habitable – mixed; targeted fixes."
      : "Not ideal – multiple risks; mitigate first.";
  const parks =
    safe.ndvi >= 0.45
      ? "Adequate greenery; preserve & add pocket parks."
      : safe.ndvi >= 0.3
      ? "Moderate greenery; corridor greening & shade trees."
      : "Low greenery; prioritize parks & streetscape planting.";
  const waste =
    safe.pop_density > 15000 && safe.industrial_km < 2
      ? "High waste pressure; deploy MRF/transfer stations & audits."
      : safe.pop_density > 15000
      ? "Elevated waste; scale collection & segregation."
      : "Standard services likely sufficient; maintain programs.";
  let risk = 0;
  if (safe.air_pm25 > 60) risk++;
  if (safe.lst_c > 34) risk++;
  if (safe.pop_density > 20000) risk++;
  if (safe.water_turbidity > 10) risk++;
  const disease =
    risk >= 3
      ? "High risk: clinics, heat shelters, vector control, potable water."
      : risk === 2
      ? "Moderate risk: monitor hotspots, shade/water points, seasonal drives."
      : "Low–moderate risk: routine surveillance & outreach.";
  return { hci, habitability, parks, waste, disease } as const;
}

function ClickCatcher({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function ClickableMap({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  const [lat, setLat] = useState(DEFAULT_POINT.lat);
  const [lon, setLon] = useState(DEFAULT_POINT.lon);

  const handlePick = (nextLat: number, nextLon: number) => {
    setLat(nextLat);
    setLon(nextLon);
    onPick(nextLat, nextLon);
  };

  return (
    <div className="space-y-3">
      <div className="rounded-2xl overflow-hidden shadow-sm border">
        <div className="w-full h-[420px] bg-muted">
          <MapContainer
            center={[lat, lon]}
            zoom={12}
            style={{ height: "100%", width: "100%" }}
            zoomControl
            dragging
            doubleClickZoom
            boxZoom
            scrollWheelZoom
          >
            <ClickCatcher onPick={handlePick} />
            <Rectangle
              bounds={[
                [19.02, 72.82],
                [19.1, 72.93],
              ]}
              pathOptions={{ color: "#ff6b6b", weight: 1, fill: true, fillOpacity: 0.1 }}
            />
            <Circle center={[lat, lon]} radius={50} pathOptions={{ color: "#111" }} />
          </MapContainer>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="col-span-2 flex items-center gap-2">
          <MapPin className="h-4 w-4" />
          <Input
            value={lat}
            onChange={(event) => handlePick(parseFloat(event.target.value || "0"), lon)}
            type="number"
            step="0.0001"
            className="h-9"
          />
          <Input
            value={lon}
            onChange={(event) => handlePick(lat, parseFloat(event.target.value || "0"))}
            type="number"
            step="0.0001"
            className="h-9"
          />
          <Button onClick={() => onPick(lat, lon)} size="sm">
            Use point
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Mumbai</Badge>
          <span className="text-sm text-muted-foreground">Click anywhere on the map</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          No internet needed — tiles disabled.
        </div>
      </div>
    </div>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  suffix = "",
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  suffix?: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-2xl border p-3">
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4" />
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <div className="font-semibold">
        {value}
        {suffix}
      </div>
    </div>
  );
}

export default function HealthyCityMumbaiApp() {
  const [point, setPoint] = useState<{ lat: number; lon: number }>(DEFAULT_POINT);

  const data: RegionData = useMemo(() => {
    try {
      return fetchRegionData(point.lat, point.lon);
    } catch (error) {
      console.warn("fetchRegionData failed", error);
      return DEFAULT_REGION;
    }
  }, [point.lat, point.lon]);

  const scores = useMemo(() => {
    try {
      return computeScores(data);
    } catch (error) {
      console.warn("computeScores failed", error);
      return computeScores(DEFAULT_REGION);
    }
  }, [data]);

  const recs = useMemo(() => {
    try {
      return recommendations(data, scores);
    } catch (error) {
      console.warn("recommendations failed", error);
      return recommendations(DEFAULT_REGION, computeScores(DEFAULT_REGION));
    }
  }, [data, scores]);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">🌆 Healthy City Index — Mumbai</h1>
          <p className="text-sm text-muted-foreground">
            Wireframe UI • Click a point → indicators, scores, and recommendations
          </p>
        </div>
        <a
          href="https://github.com/akkiannu/healthy-city-index"
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-10 items-center gap-2 rounded-md border border-slate-200 bg-white px-4 text-sm font-medium transition hover:bg-slate-100"
        >
          <Github className="h-4 w-4" /> Repo
        </a>
      </header>

      <Tabs defaultValue="map" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="map">🗺️ Map Explorer</TabsTrigger>
          <TabsTrigger value="ai">🧠 AI Recommendations</TabsTrigger>
        </TabsList>

        <TabsContent value="map" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <Card className="lg:col-span-7">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="h-5 w-5" /> Pick a location
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ClickableMap onPick={(lat, lon) => setPoint({ lat, lon })} />
              </CardContent>
            </Card>

            <Card className="lg:col-span-5">
              <CardHeader>
                <CardTitle>
                  📊 Indicators @ {Number(data?.lat ?? DEFAULT_REGION.lat).toFixed(4)}, {" "}
                  {Number(data?.lon ?? DEFAULT_REGION.lon).toFixed(4)}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <Metric icon={Cloud} label="PM2.5" value={data?.air_pm25 ?? DEFAULT_REGION.air_pm25} suffix=" μg/m³" />
                  <Metric icon={Cloud} label="NO₂" value={data?.air_no2 ?? DEFAULT_REGION.air_no2} suffix=" μg/m³" />
                  <Metric
                    icon={Droplets}
                    label="Water turbidity"
                    value={data?.water_turbidity ?? DEFAULT_REGION.water_turbidity}
                    suffix=" NTU"
                  />
                  <Metric icon={Trees} label="NDVI" value={data?.ndvi ?? DEFAULT_REGION.ndvi} />
                  <Metric
                    icon={Users}
                    label="Population dens."
                    value={data?.pop_density ?? DEFAULT_REGION.pop_density}
                    suffix=" /km²"
                  />
                  <Metric
                    icon={Thermometer}
                    label="LST"
                    value={data?.lst_c ?? DEFAULT_REGION.lst_c}
                    suffix=" °C"
                  />
                  <Metric
                    icon={Factory}
                    label="Industrial proximity"
                    value={data?.industrial_km ?? DEFAULT_REGION.industrial_km}
                    suffix=" km"
                  />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-center">
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Composite HCI</div>
                    <div className="text-4xl font-semibold">{composite(scores).toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground">(0–1, higher is better)</div>
                  </div>
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart
                        data={[
                          { metric: "Air", v: +(scores.air * 100).toFixed(0) },
                          { metric: "Water", v: +(scores.water * 100).toFixed(0) },
                          { metric: "Green", v: +(scores.green * 100).toFixed(0) },
                          { metric: "Population", v: +(scores.population * 100).toFixed(0) },
                          { metric: "Temp", v: +(scores.temperature * 100).toFixed(0) },
                          { metric: "Industrial", v: +(scores.industrial * 100).toFixed(0) },
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius="75%"
                      >
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis tick={false} axisLine={false} />
                        <Radar name="Score" dataKey="v" fill="#0f172a" fillOpacity={0.3} stroke="#0f172a" />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  ⚠️ Mock values for wireframe. Replace with API calls to your Python backend.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="ai" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <Card className="lg:col-span-7">
              <CardHeader>
                <CardTitle>LLM Recommendations (stub)</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <Metric icon={MapPin} label="Lat" value={(point.lat ?? DEFAULT_POINT.lat).toFixed(5)} />
                  <Metric icon={MapPin} label="Lon" value={(point.lon ?? DEFAULT_POINT.lon).toFixed(5)} />
                  <Metric icon={Cloud} label="PM2.5" value={data?.air_pm25 ?? DEFAULT_REGION.air_pm25} suffix=" μg/m³" />
                  <Metric icon={Thermometer} label="LST" value={data?.lst_c ?? DEFAULT_REGION.lst_c} suffix=" °C" />
                  <Metric icon={Users} label="Pop. dens." value={data?.pop_density ?? DEFAULT_REGION.pop_density} suffix=" /km²" />
                  <Metric icon={Droplets} label="Turbidity" value={data?.water_turbidity ?? DEFAULT_REGION.water_turbidity} suffix=" NTU" />
                </div>

                <div className="rounded-xl border p-4 space-y-2">
                  <div className="text-sm text-muted-foreground">Composite HCI</div>
                  <div className="text-3xl font-semibold">{recs.hci.toFixed(2)}</div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-muted-foreground mb-1">Habitability</div>
                    <div className="font-medium">{recs.habitability}</div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-muted-foreground mb-1">Parks / Greenery</div>
                    <div className="font-medium">{recs.parks}</div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-muted-foreground mb-1">Waste Management</div>
                    <div className="font-medium">{recs.waste}</div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-xs text-muted-foreground mb-1">Disease Spread Risk</div>
                    <div className="font-medium">{recs.disease}</div>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  ℹ️ Replace with a real LLM later; send this panel the raw indicators & scores as structured context.
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-5">
              <CardHeader>
                <CardTitle>Backend wiring (TODO)</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <ol className="list-decimal pl-5 space-y-2">
                  <li>
                    Expose a Python endpoint <code>/api/region?lat=&amp;lon=</code> returning the 6 indicators.
                  </li>
                  <li>
                    Swap <code>fetchRegionData</code> to call the endpoint; handle loading &amp; errors.
                  </li>
                  <li>
                    Optionally add a TileLayer (OSM) when you want a basemap (will require network).
                  </li>
                  <li>Preload Mumbai vectors (industrial polygons) for visual overlays.</li>
                </ol>
                <div className="rounded-xl bg-muted p-3 text-muted-foreground">
                  Tip: keep the JSON shape identical to <code>RegionData</code> for a drop-in replacement.
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

if (typeof window !== "undefined") {
  try {
    console.assert(minmax(10, 0, 10) === 1, "minmax upper bound should be 1");
    console.assert(minmax(0, 0, 10) === 0, "minmax lower bound should be 0");
    const dummy: RegionData = {
      lat: 0,
      lon: 0,
      air_pm25: 50,
      air_no2: 30,
      water_turbidity: 5,
      ndvi: 0.4,
      pop_density: 10000,
      lst_c: 32,
      industrial_km: 3,
    };
    const s = computeScores(dummy);
    const h = composite(s);
    console.assert(h >= 0 && h <= 1, "composite should be within [0,1]");
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn("Runtime tests failed:", error);
  }
}
