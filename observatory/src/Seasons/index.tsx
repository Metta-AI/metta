import { FC, useContext, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { AppContext } from '../AppContext'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'

export const SeasonsPage: FC = () => {
  const { repo } = useContext(AppContext)
  const [seasons, setSeasons] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    const load = async () => {
      try {
        const data = await repo.getSeasons()
        if (!ignore) {
          setSeasons(data)
          setError(null)
        }
      } catch (err: any) {
        if (!ignore) {
          setError(err.message)
        }
      } finally {
        if (!ignore) {
          setLoading(false)
        }
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [repo])

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <Card title="Seasons">
        {loading ? (
          <div className="flex justify-center py-8">
            <Spinner size="lg" />
          </div>
        ) : error ? (
          <div className="text-red-600 py-4">{error}</div>
        ) : seasons.length === 0 ? (
          <div className="text-gray-500 py-4">No seasons found</div>
        ) : (
          <div className="space-y-2">
            {seasons.map((season) => (
              <Link
                key={season}
                to={`/seasons/${season}`}
                className="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <span className="text-lg font-medium text-gray-900">{season}</span>
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
